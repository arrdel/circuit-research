"""Publication-ready temporal/layer causal heatmaps and counterfactual grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def plot_layer_timestep_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    step_labels: list[str],
    title: str,
    cbar_label: str,
    out_path: str | Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> Path:
    """Plot a Layer x Timestep-window causal map and save to out_path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6.0, len(step_labels) * 1.1), max(5.0, len(layers) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Denoising Step Window")
    ax.set_ylabel("DiT Layer")
    ax.set_title(title)

    # Annotate values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.55 * (vmax - vmin) + vmin else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_counterfactual_grid(
    rows: list[tuple[str, list[tuple[str, Image.Image]]]],
    out_path: str | Path,
    tile_size: int = 256,
) -> Path:
    """Save a labeled grid of (Clean, Corrupt, Patched, Controls...) images."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot save empty grid.")

    n_cols = max(len(cols) for _, cols in rows)
    header_h = 28
    pad = 8
    w = n_cols * (tile_size + pad) + pad
    h = len(rows) * (tile_size + header_h + pad) + pad
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for r_idx, (row_label, cols) in enumerate(rows):
        y0 = pad + r_idx * (tile_size + header_h + pad)
        for c_idx, (col_label, img) in enumerate(cols):
            x0 = pad + c_idx * (tile_size + pad)
            label = f"{row_label} | {col_label}" if c_idx == 0 else col_label
            draw.text((x0 + 2, y0 + 4), label[:42], fill=(20, 20, 20))
            resized = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x0, y0 + header_h))

    canvas.save(out)
    return out
