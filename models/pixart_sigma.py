"""PixArt-Sigma model adapter with prompt embedding caching and hookable denoising."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtTransformer2DModel
from diffusers.image_processor import PixArtImageProcessor
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
DEFAULT_BASE_COMPONENTS_ID = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"


class StepCallback(Protocol):
    def on_step_start(self, step_idx: int, timestep: torch.Tensor) -> None: ...
    def on_step_end(self, step_idx: int, timestep: torch.Tensor, latents: torch.Tensor) -> None: ...


@dataclass
class EncodedPrompt:
    prompt: str
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    token_strings: list[str]


@dataclass
class GenerationResult:
    prompt: str
    seed: int
    image: Image.Image
    latents: torch.Tensor
    timesteps: list[float]


def _prompt_cache_key(model_id: str, prompt: str, max_sequence_length: int) -> str:
    payload = json.dumps(
        {"model_id": model_id, "prompt": prompt, "max_len": max_sequence_length},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class PixArtSigmaRunner:
    """Memory-conscious PixArt-Sigma runner supporting cached T5 embeddings and hooks."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        base_components_id: str | None = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | Path = "artifacts/cache/prompt_embeds",
        max_sequence_length: int = 300,
    ) -> None:
        self.model_id = model_id
        if base_components_id is not None:
            self.base_components_id = base_components_id
        elif model_id == "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS":
            self.base_components_id = model_id
        else:
            self.base_components_id = DEFAULT_BASE_COMPONENTS_ID
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_sequence_length = max_sequence_length

        self.transformer: PixArtTransformer2DModel | None = None
        self.vae: AutoencoderKL | None = None
        self.scheduler: DPMSolverMultistepScheduler | None = None
        self.image_processor: PixArtImageProcessor | None = None
        self.vae_scale_factor: int = 8

    @property
    def num_layers(self) -> int:
        if self.transformer is None:
            raise RuntimeError("Call load_denoiser() first.")
        return len(self.transformer.transformer_blocks)

    @property
    def num_heads(self) -> int:
        if self.transformer is None:
            raise RuntimeError("Call load_denoiser() first.")
        return int(self.transformer.config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        if self.transformer is None:
            raise RuntimeError("Call load_denoiser() first.")
        return int(self.transformer.config.attention_head_dim)

    def encode_and_cache_prompts(
        self,
        prompts: list[str],
        text_encoder_device_map: str | dict | None = "auto",
        max_memory_per_gpu: str | None = "5500MiB",
        batch_size: int = 16,
        force_recompute: bool = False,
    ) -> dict[str, EncodedPrompt]:
        """Encode prompts (plus empty negative prompt) using T5-XXL and cache on disk.

        T5-XXL is loaded only if at least one prompt is missing from cache, and is
        immediately released afterward so denoiser VRAM stays below ~2.5 GiB.
        """
        unique_prompts = list(dict.fromkeys([""] + list(prompts)))
        results: dict[str, EncodedPrompt] = {}
        missing: list[str] = []

        for p in unique_prompts:
            cache_file = self.cache_dir / f"{_prompt_cache_key(self.model_id, p, self.max_sequence_length)}.pt"
            if cache_file.exists() and not force_recompute:
                blob = torch.load(cache_file, map_location="cpu", weights_only=True)
                results[p] = EncodedPrompt(
                    prompt=p,
                    prompt_embeds=blob["prompt_embeds"],
                    prompt_attention_mask=blob["prompt_attention_mask"],
                    token_strings=blob["token_strings"],
                )
            else:
                missing.append(p)

        if not missing:
            return results

        tokenizer = T5Tokenizer.from_pretrained(self.base_components_id, subfolder="tokenizer")
        load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if text_encoder_device_map is not None:
            load_kwargs["device_map"] = text_encoder_device_map
            if max_memory_per_gpu is not None and torch.cuda.device_count() > 0:
                n_gpus = torch.cuda.device_count()
                load_kwargs["max_memory"] = {i: max_memory_per_gpu for i in range(n_gpus)}
                load_kwargs["max_memory"]["cpu"] = "64GiB"
        text_encoder = T5EncoderModel.from_pretrained(
            self.base_components_id, subfolder="text_encoder", **load_kwargs
        )
        text_encoder.eval()

        # Determine device of first embedding parameter for input_ids placement
        enc_device = next(text_encoder.parameters()).device

        with torch.no_grad():
            for start in range(0, len(missing), batch_size):
                chunk = missing[start : start + batch_size]
                tokenized = tokenizer(
                    chunk,
                    padding="max_length",
                    max_length=self.max_sequence_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(enc_device)
                attn_mask = tokenized.attention_mask.to(enc_device)
                embeds = text_encoder(input_ids, attention_mask=attn_mask)[0].cpu().to(self.dtype)
                attn_mask_cpu = tokenized.attention_mask.cpu()

                for i, p in enumerate(chunk):
                    ids_i = tokenized.input_ids[i].tolist()
                    mask_i = attn_mask_cpu[i].tolist()
                    toks = [
                        tokenizer.convert_ids_to_tokens(tid)
                        for tid, m in zip(ids_i, mask_i)
                        if m == 1
                    ]
                    ep = EncodedPrompt(
                        prompt=p,
                        prompt_embeds=embeds[i : i + 1],
                        prompt_attention_mask=attn_mask_cpu[i : i + 1],
                        token_strings=toks,
                    )
                    cache_file = (
                        self.cache_dir
                        / f"{_prompt_cache_key(self.model_id, p, self.max_sequence_length)}.pt"
                    )
                    torch.save(
                        {
                            "prompt_embeds": ep.prompt_embeds,
                            "prompt_attention_mask": ep.prompt_attention_mask,
                            "token_strings": ep.token_strings,
                        },
                        cache_file,
                    )
                    results[p] = ep

        del text_encoder
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def load_denoiser(self) -> None:
        """Load transformer, VAE, and scheduler onto self.device."""
        if self.transformer is not None:
            return
        self.transformer = PixArtTransformer2DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=self.dtype
        ).to(self.device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self.base_components_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            self.base_components_id, subfolder="scheduler"
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def generate(
        self,
        prompt_entry: EncodedPrompt,
        negative_entry: EncodedPrompt,
        seed: int = 42,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        height: int | None = None,
        width: int | None = None,
        step_callbacks: list[StepCallback] | None = None,
        decode_image: bool = True,
    ) -> GenerationResult:
        """Run deterministic PixArt-Sigma denoising with optional step callbacks."""
        self.load_denoiser()
        assert self.transformer is not None
        assert self.vae is not None
        assert self.scheduler is not None
        assert self.image_processor is not None

        default_hw = int(self.transformer.config.sample_size) * self.vae_scale_factor
        req_h = height or default_hw
        req_w = width or default_hw
        if self.transformer.config.sample_size == 128:
            bin_map = ASPECT_RATIO_1024_BIN
        else:
            bin_map = ASPECT_RATIO_512_BIN
        bin_h, bin_w = self.image_processor.classify_height_width_bin(req_h, req_w, ratios=bin_map)

        do_cfg = guidance_scale > 1.0
        pos_embeds = prompt_entry.prompt_embeds.to(device=self.device, dtype=self.dtype)
        pos_mask = prompt_entry.prompt_attention_mask.to(device=self.device)

        if do_cfg:
            neg_embeds = negative_entry.prompt_embeds.to(device=self.device, dtype=self.dtype)
            neg_mask = negative_entry.prompt_attention_mask.to(device=self.device)
            encoder_hidden_states = torch.cat([neg_embeds, pos_embeds], dim=0)
            encoder_attention_mask = torch.cat([neg_mask, pos_mask], dim=0)
        else:
            encoder_hidden_states = pos_embeds
            encoder_attention_mask = pos_mask

        # Fresh scheduler state per generation so DPM-Solver multistep buffer is clean
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latent_channels = int(self.transformer.config.in_channels)
        latent_shape = (
            1,
            latent_channels,
            int(bin_h) // self.vae_scale_factor,
            int(bin_w) // self.vae_scale_factor,
        )
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(latent_shape, generator=gen, device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if self.transformer.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor([[bin_h, bin_w]], device=self.device, dtype=self.dtype).repeat(
                    2 if do_cfg else 1, 1
                ),
                "aspect_ratio": torch.tensor([[bin_h / bin_w]], device=self.device, dtype=self.dtype).repeat(
                    2 if do_cfg else 1, 1
                ),
            }
        else:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        callbacks = step_callbacks or []
        recorded_t: list[float] = []

        for step_idx, t in enumerate(timesteps):
            for cb in callbacks:
                cb.on_step_start(step_idx, t)

            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_t = t
            if not torch.is_tensor(current_t):
                current_t = torch.tensor([current_t], device=self.device, dtype=torch.int64)
            elif current_t.ndim == 0:
                current_t = current_t[None].to(self.device)
            current_t = current_t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=current_t,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            recorded_t.append(float(t.item() if torch.is_tensor(t) else t))

            for cb in callbacks:
                cb.on_step_end(step_idx, t, latents)

        if decode_image:
            decoded = self.vae.decode(
                latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False
            )[0]
            decoded = self.image_processor.resize_and_crop_tensor(decoded, req_w, req_h)
            pil_image = self.image_processor.postprocess(decoded, output_type="pil")[0]
        else:
            pil_image = Image.new("RGB", (req_w, req_h))

        return GenerationResult(
            prompt=prompt_entry.prompt,
            seed=seed,
            image=pil_image,
            latents=latents.detach().cpu(),
            timesteps=recorded_t,
        )
