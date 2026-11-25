"""Causal intervention primitives (activation patching, ablation, and controls) for DiTs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from diffusers import PixArtTransformer2DModel

from instrumentation.hooks import ActivationCache, BranchMode, HookPoint


InterventionKind = Literal[
    "patch",
    "interpolate",
    "zero_ablate",
    "mean_ablate",
    "random_norm_matched",
]


@dataclass
class InterventionSpec:
    """Declarative specification for a causal intervention during denoising."""

    kind: InterventionKind
    hook_point: HookPoint
    layers: list[int]
    steps: list[int]
    source_cache: ActivationCache | None = None
    heads: list[int] | None = None
    token_indices: list[int] | None = None
    branch: BranchMode = "cond"
    alpha: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind in ("patch", "interpolate", "mean_ablate") and self.source_cache is None:
            raise ValueError(f"Intervention kind='{self.kind}' requires source_cache.")


def _apply_tensor_edit(
    target: torch.Tensor,
    spec: InterventionSpec,
    step_idx: int,
    layer_idx: int,
    num_heads: int,
) -> torch.Tensor:
    """Return modified activation tensor without mutating unrelated batch branches."""
    out = target.clone()

    # Determine batch slice indices
    if spec.branch == "both" or out.shape[0] < 2 or out.shape[0] % 2 != 0:
        b_slice = slice(None)
    elif spec.branch == "cond":
        b_slice = slice(out.shape[0] // 2, out.shape[0])
    else:
        b_slice = slice(0, out.shape[0] // 2)

    sub = out[b_slice]  # (B_sub, N, D)

    # Head-level reshaping for pre-output projection hooks
    is_head_hook = spec.hook_point in ("attn1_head_pre_proj", "attn2_head_pre_proj")
    if is_head_hook and spec.heads is not None:
        bs, seq_len, inner_dim = sub.shape
        head_dim = inner_dim // num_heads
        sub_view = sub.view(bs, seq_len, num_heads, head_dim)
        if spec.token_indices is not None:
            tok_idx = torch.tensor(spec.token_indices, device=sub.device, dtype=torch.long)
            head_idx = torch.tensor(spec.heads, device=sub.device, dtype=torch.long)
            cur = sub_view[:, tok_idx][:, :, head_idx]
        else:
            head_idx = torch.tensor(spec.heads, device=sub.device, dtype=torch.long)
            cur = sub_view[:, :, head_idx]
    else:
        if spec.token_indices is not None:
            tok_idx = torch.tensor(spec.token_indices, device=sub.device, dtype=torch.long)
            cur = sub[:, tok_idx]
        else:
            cur = sub

    # Compute replacement tensor matching cur.shape
    if spec.kind == "zero_ablate":
        new_val = torch.zeros_like(cur)
    elif spec.kind == "random_norm_matched":
        gen = torch.Generator(device=cur.device).manual_seed(
            spec.seed + step_idx * 1009 + layer_idx * 37
        )
        rand = torch.randn(cur.shape, generator=gen, device=cur.device, dtype=torch.float32)
        rand_norm = rand.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cur_norm = cur.float().norm(dim=-1, keepdim=True)
        new_val = (rand / rand_norm * cur_norm).to(cur.dtype)
    else:
        assert spec.source_cache is not None
        src_full = spec.source_cache.get(step_idx, layer_idx, spec.hook_point).to(
            device=cur.device, dtype=cur.dtype
        )
        if src_full.shape[0] != sub.shape[0]:
            # If cache stored single cond slice and sub is both, expand
            if src_full.shape[0] == 1 and sub.shape[0] > 1:
                src_full = src_full.expand(sub.shape[0], -1, -1)
        if is_head_hook and spec.heads is not None:
            bs, seq_len, inner_dim = src_full.shape
            head_dim = inner_dim // num_heads
            src_view = src_full.view(bs, seq_len, num_heads, head_dim)
            if spec.token_indices is not None:
                src_slice = src_view[:, tok_idx][:, :, head_idx]
            else:
                src_slice = src_view[:, :, head_idx]
        else:
            if spec.token_indices is not None:
                src_slice = src_full[:, tok_idx]
            else:
                src_slice = src_full

        if spec.kind in ("patch", "mean_ablate"):
            new_val = src_slice
        elif spec.kind == "interpolate":
            new_val = (1.0 - spec.alpha) * cur + spec.alpha * src_slice
        else:
            raise ValueError(f"Unsupported intervention kind: {spec.kind}")

    # Write back into sub / out
    if is_head_hook and spec.heads is not None:
        if spec.token_indices is not None:
            for ti_pos, ti in enumerate(spec.token_indices):
                for hi_pos, hi in enumerate(spec.heads):
                    sub_view[:, ti, hi] = new_val[:, ti_pos, hi_pos]
        else:
            sub_view[:, :, head_idx] = new_val
        out[b_slice] = sub_view.reshape(sub.shape)
    else:
        if spec.token_indices is not None:
            sub[:, tok_idx] = new_val
        else:
            sub = new_val
        out[b_slice] = sub

    return out


class InterventionController:
    """Applies one or more InterventionSpecs during PixArtTransformer2DModel denoising."""

    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        specs: list[InterventionSpec],
    ) -> None:
        self.transformer = transformer
        self.specs = specs
        self.num_heads = int(transformer.config.num_attention_heads)
        self._current_step: int | None = None
        self._handles: list = []

    def on_step_start(self, step_idx: int, timestep: torch.Tensor) -> None:
        self._current_step = step_idx

    def on_step_end(self, step_idx: int, timestep: torch.Tensor, latents: torch.Tensor) -> None:
        self._current_step = None

    def _active_specs(self, layer_idx: int, hook_point: HookPoint) -> list[InterventionSpec]:
        if self._current_step is None:
            return []
        return [
            s
            for s in self.specs
            if s.hook_point == hook_point
            and layer_idx in s.layers
            and self._current_step in s.steps
        ]

    def _attach(self) -> None:
        needed_pairs = {
            (layer_idx, spec.hook_point)
            for spec in self.specs
            for layer_idx in spec.layers
        }

        for layer_idx, hook_point in sorted(needed_pairs):
            block = self.transformer.transformer_blocks[layer_idx]

            if hook_point == "block_out":
                def make_block_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        specs = self._active_specs(idx, "block_out")
                        if not specs:
                            return out
                        assert self._current_step is not None
                        is_tuple = isinstance(out, tuple)
                        t = out[0] if is_tuple else out
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return (t,) + out[1:] if is_tuple else t
                    return _hook
                self._handles.append(block.register_forward_hook(make_block_hook(layer_idx)))

            elif hook_point == "attn1_out":
                def make_attn1_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        specs = self._active_specs(idx, "attn1_out")
                        if not specs:
                            return out
                        assert self._current_step is not None
                        t = out
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return t
                    return _hook
                self._handles.append(block.attn1.register_forward_hook(make_attn1_hook(layer_idx)))

            elif hook_point == "attn2_out" and block.attn2 is not None:
                def make_attn2_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        specs = self._active_specs(idx, "attn2_out")
                        if not specs:
                            return out
                        assert self._current_step is not None
                        t = out
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return t
                    return _hook
                self._handles.append(block.attn2.register_forward_hook(make_attn2_hook(layer_idx)))

            elif hook_point == "ff_out":
                def make_ff_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        specs = self._active_specs(idx, "ff_out")
                        if not specs:
                            return out
                        assert self._current_step is not None
                        t = out
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return t
                    return _hook
                self._handles.append(block.ff.register_forward_hook(make_ff_hook(layer_idx)))

            elif hook_point == "attn1_head_pre_proj":
                def make_attn1_pre_hook(idx: int):
                    def _hook(_mod, args):
                        specs = self._active_specs(idx, "attn1_head_pre_proj")
                        if not specs:
                            return args
                        assert self._current_step is not None
                        t = args[0]
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return (t,) + args[1:]
                    return _hook
                self._handles.append(
                    block.attn1.to_out[0].register_forward_pre_hook(make_attn1_pre_hook(layer_idx))
                )

            elif hook_point == "attn2_head_pre_proj" and block.attn2 is not None:
                def make_attn2_pre_hook(idx: int):
                    def _hook(_mod, args):
                        specs = self._active_specs(idx, "attn2_head_pre_proj")
                        if not specs:
                            return args
                        assert self._current_step is not None
                        t = args[0]
                        for sp in specs:
                            t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                        return (t,) + args[1:]
                    return _hook
                self._handles.append(
                    block.attn2.to_out[0].register_forward_pre_hook(make_attn2_pre_hook(layer_idx))
                )

            elif hook_point == "attn2_in_txt" and block.attn2 is not None:
                def make_txt_pre_hook(idx: int):
                    def _hook(_mod, args, kwargs):
                        specs = self._active_specs(idx, "attn2_in_txt")
                        if not specs:
                            return args, kwargs
                        assert self._current_step is not None
                        new_kwargs = dict(kwargs)
                        if "encoder_hidden_states" in new_kwargs and new_kwargs["encoder_hidden_states"] is not None:
                            t = new_kwargs["encoder_hidden_states"]
                            for sp in specs:
                                t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                            new_kwargs["encoder_hidden_states"] = t
                            return args, new_kwargs
                        if len(args) > 1 and args[1] is not None:
                            t = args[1]
                            for sp in specs:
                                t = _apply_tensor_edit(t, sp, self._current_step, idx, self.num_heads)
                            new_args = (args[0], t) + args[2:]
                            return new_args, kwargs
                        return args, kwargs
                    return _hook
                self._handles.append(
                    block.attn2.register_forward_pre_hook(make_txt_pre_hook(layer_idx), with_kwargs=True)
                )

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._current_step = None

    @contextmanager
    def active(self) -> Iterator[InterventionController]:
        self._attach()
        try:
            yield self
        finally:
            self._detach()
