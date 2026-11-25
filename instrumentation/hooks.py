"""Activation capture and cross-attention instrumentation for Diffusion Transformers."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Literal

import torch
import torch.nn.functional as F
from diffusers import PixArtTransformer2DModel
from diffusers.models.attention_processor import Attention


HookPoint = Literal[
    "block_out",
    "attn1_out",
    "attn2_out",
    "ff_out",
    "attn1_head_pre_proj",
    "attn2_head_pre_proj",
    "attn2_in_txt",
]

BranchMode = Literal["cond", "uncond", "both"]


def extract_branch_slice(tensor: torch.Tensor, branch: BranchMode = "cond") -> torch.Tensor:
    """Slice conditional/unconditional branch from a CFG-batched tensor."""
    if branch == "both" or tensor.shape[0] < 2 or tensor.shape[0] % 2 != 0:
        return tensor
    half = tensor.shape[0] // 2
    if branch == "cond":
        return tensor[half:]
    return tensor[:half]


class ActivationCache:
    """Stores activations keyed by (step_idx, layer_idx, hook_point)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, int, str], torch.Tensor] = {}
        self.attn_probs: dict[tuple[int, int], torch.Tensor] = {}

    def set(self, step_idx: int, layer_idx: int, hook_point: str, tensor: torch.Tensor) -> None:
        self._store[(step_idx, layer_idx, hook_point)] = tensor

    def get(self, step_idx: int, layer_idx: int, hook_point: str) -> torch.Tensor:
        return self._store[(step_idx, layer_idx, hook_point)]

    def has(self, step_idx: int, layer_idx: int, hook_point: str) -> bool:
        return (step_idx, layer_idx, hook_point) in self._store

    def keys(self) -> list[tuple[int, int, str]]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()
        self.attn_probs.clear()

    @classmethod
    def average(cls, caches: list[ActivationCache]) -> ActivationCache:
        """Compute elementwise mean across a list of matched ActivationCaches."""
        if not caches:
            raise ValueError("Cannot average an empty list of caches.")
        out = cls()
        accum: dict[tuple[int, int, str], torch.Tensor] = {}
        counts: dict[tuple[int, int, str], int] = defaultdict(int)
        for cache in caches:
            for key, val in cache._store.items():
                v_fp32 = val.float()
                if key not in accum:
                    accum[key] = v_fp32.clone()
                else:
                    accum[key].add_(v_fp32)
                counts[key] += 1
        for key, total in accum.items():
            ref_dtype = caches[0]._store[key].dtype
            out._store[key] = (total / counts[key]).to(ref_dtype)
        return out


class CrossAttentionCaptureProcessor:
    """Drop-in AttnProcessor that captures cross-attention probabilities when active."""

    def __init__(
        self,
        layer_idx: int,
        cache: ActivationCache,
        step_getter,
        capture_steps: set[int] | None = None,
        branch: BranchMode = "cond",
        store_cpu: bool = True,
    ) -> None:
        self.layer_idx = layer_idx
        self.cache = cache
        self.step_getter = step_getter
        self.capture_steps = capture_steps
        self.branch = branch
        self.store_cpu = store_cpu

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        step_idx = self.step_getter()
        should_capture = (
            step_idx is not None
            and (self.capture_steps is None or step_idx in self.capture_steps)
        )

        if should_capture:
            scale = head_dim**-0.5
            attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask.float()
            attn_probs = torch.softmax(attn_scores, dim=-1)
            to_save = extract_branch_slice(attn_probs.to(hidden_states.dtype), self.branch).detach()
            if self.store_cpu:
                to_save = to_save.cpu()
            self.cache.attn_probs[(step_idx, self.layer_idx)] = to_save
            hidden_states = torch.matmul(attn_probs.to(value.dtype), value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class DiTHookManager:
    """Step-aware activation capture for PixArtTransformer2DModel."""

    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        layers: list[int] | None = None,
        hook_points: list[HookPoint] | None = None,
        steps: list[int] | None = None,
        branch: BranchMode = "cond",
        store_cpu: bool = True,
        capture_attn2_probs: bool = False,
    ) -> None:
        self.transformer = transformer
        num_layers = len(transformer.transformer_blocks)
        self.layers = sorted(layers) if layers is not None else list(range(num_layers))
        self.hook_points: list[HookPoint] = hook_points or ["block_out"]
        self.steps: set[int] | None = set(steps) if steps is not None else None
        self.branch: BranchMode = branch
        self.store_cpu = store_cpu
        self.capture_attn2_probs = capture_attn2_probs

        self.cache = ActivationCache()
        self._current_step: int | None = None
        self._handles: list = []
        self._orig_attn_processors: dict | None = None

    def on_step_start(self, step_idx: int, timestep: torch.Tensor) -> None:
        self._current_step = step_idx

    def on_step_end(self, step_idx: int, timestep: torch.Tensor, latents: torch.Tensor) -> None:
        self._current_step = None

    def _should_record(self) -> bool:
        if self._current_step is None:
            return False
        if self.steps is not None and self._current_step not in self.steps:
            return False
        return True

    def _save_tensor(self, layer_idx: int, hook_point: str, tensor: torch.Tensor) -> None:
        assert self._current_step is not None
        sliced = extract_branch_slice(tensor, self.branch).detach()
        if self.store_cpu:
            sliced = sliced.cpu()
        else:
            sliced = sliced.clone()
        self.cache.set(self._current_step, layer_idx, hook_point, sliced)

    def _attach(self) -> None:
        for layer_idx in self.layers:
            block = self.transformer.transformer_blocks[layer_idx]

            if "block_out" in self.hook_points:
                def make_block_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        if self._should_record():
                            t = out[0] if isinstance(out, tuple) else out
                            self._save_tensor(idx, "block_out", t)
                    return _hook
                self._handles.append(block.register_forward_hook(make_block_hook(layer_idx)))

            if "attn1_out" in self.hook_points:
                def make_attn1_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        if self._should_record():
                            self._save_tensor(idx, "attn1_out", out)
                    return _hook
                self._handles.append(block.attn1.register_forward_hook(make_attn1_hook(layer_idx)))

            if "attn2_out" in self.hook_points and block.attn2 is not None:
                def make_attn2_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        if self._should_record():
                            self._save_tensor(idx, "attn2_out", out)
                    return _hook
                self._handles.append(block.attn2.register_forward_hook(make_attn2_hook(layer_idx)))

            if "ff_out" in self.hook_points:
                def make_ff_hook(idx: int):
                    def _hook(_mod, _inp, out):
                        if self._should_record():
                            self._save_tensor(idx, "ff_out", out)
                    return _hook
                self._handles.append(block.ff.register_forward_hook(make_ff_hook(layer_idx)))

            if "attn1_head_pre_proj" in self.hook_points:
                def make_attn1_pre_hook(idx: int):
                    def _hook(_mod, args):
                        if self._should_record():
                            self._save_tensor(idx, "attn1_head_pre_proj", args[0])
                    return _hook
                self._handles.append(
                    block.attn1.to_out[0].register_forward_pre_hook(make_attn1_pre_hook(layer_idx))
                )

            if "attn2_head_pre_proj" in self.hook_points and block.attn2 is not None:
                def make_attn2_pre_hook(idx: int):
                    def _hook(_mod, args):
                        if self._should_record():
                            self._save_tensor(idx, "attn2_head_pre_proj", args[0])
                    return _hook
                self._handles.append(
                    block.attn2.to_out[0].register_forward_pre_hook(make_attn2_pre_hook(layer_idx))
                )

            if "attn2_in_txt" in self.hook_points and block.attn2 is not None:
                def make_txt_hook(idx: int):
                    def _hook(_mod, args, kwargs):
                        if self._should_record():
                            enc = kwargs.get("encoder_hidden_states")
                            if enc is None and len(args) > 1:
                                enc = args[1]
                            if enc is not None:
                                self._save_tensor(idx, "attn2_in_txt", enc)
                    return _hook
                self._handles.append(
                    block.attn2.register_forward_pre_hook(make_txt_hook(layer_idx), with_kwargs=True)
                )

        if self.capture_attn2_probs:
            self._orig_attn_processors = dict(self.transformer.attn_processors)
            new_procs = dict(self._orig_attn_processors)
            for layer_idx in self.layers:
                key = f"transformer_blocks.{layer_idx}.attn2.processor"
                if key in new_procs:
                    new_procs[key] = CrossAttentionCaptureProcessor(
                        layer_idx=layer_idx,
                        cache=self.cache,
                        step_getter=lambda: self._current_step,
                        capture_steps=self.steps,
                        branch=self.branch,
                        store_cpu=self.store_cpu,
                    )
            self.transformer.set_attn_processor(new_procs)

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._orig_attn_processors is not None:
            self.transformer.set_attn_processor(self._orig_attn_processors)
            self._orig_attn_processors = None
        self._current_step = None

    @contextmanager
    def active(self, clear_cache: bool = True) -> Iterator[ActivationCache]:
        if clear_cache:
            self.cache.clear()
        self._attach()
        try:
            yield self.cache
        finally:
            self._detach()
