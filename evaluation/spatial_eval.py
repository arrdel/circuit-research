"""Automated spatial-relation and color-attribute-binding evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SIGLIP_ID = "google/siglip-base-patch16-384"


@dataclass
class DetectionBox:
    label: str
    score: float
    x1: float  # normalized [0, 1]
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class ImageEvalResult:
    box_a: DetectionBox | None
    box_b: DetectionBox | None
    both_detected: bool
    dx: float | None  # cx_b - cx_a (>0 means A is left of B)
    relation_pred: str  # "left" | "right" | "unknown"
    relation_correct: bool | None
    color_a_margin: float | None  # sim(crop_a, color_a) - sim(crop_a, color_b)
    color_b_margin: float | None  # sim(crop_b, color_b) - sim(crop_b, color_a)
    color_binding_correct: bool | None
    siglip_embedding: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("siglip_embedding", None)
        return d


def _box_iou(a: DetectionBox, b: DetectionBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 1e-8 else 0.0


def _match_label(pred_label: str, target_obj: str) -> bool:
    p = pred_label.lower().strip()
    t = target_obj.lower().strip()
    return t in p or p in t


class SpatialColorEvaluator:
    """Evaluates spatial left/right relations and object-color binding using GroundingDINO + SigLIP."""

    def __init__(
        self,
        detector_id: str = DEFAULT_DETECTOR_ID,
        siglip_id: str = DEFAULT_SIGLIP_ID,
        device: str = "cuda:0",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        spatial_margin: float = 0.04,
    ) -> None:
        self.detector_id = detector_id
        self.siglip_id = siglip_id
        self.device = torch.device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.spatial_margin = spatial_margin

        self._det_processor = None
        self._det_model = None
        self._sig_processor = None
        self._sig_model = None

    def load(self) -> None:
        if self._det_model is None:
            self._det_processor = AutoProcessor.from_pretrained(self.detector_id, use_fast=False)
            self._det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.detector_id
            ).to(self.device)
            self._det_model.eval()
            self._det_model.requires_grad_(False)

        if self._sig_model is None:
            self._sig_processor = AutoProcessor.from_pretrained(self.siglip_id, use_fast=False)
            self._sig_model = AutoModel.from_pretrained(self.siglip_id).to(self.device)
            self._sig_model.eval()
            self._sig_model.requires_grad_(False)

    @torch.no_grad()
    def detect_two_objects(
        self, image: Image.Image, obj_a: str, obj_b: str
    ) -> tuple[DetectionBox | None, DetectionBox | None]:
        self.load()
        assert self._det_processor is not None and self._det_model is not None

        w, h = image.size
        caption = f"{obj_a} . {obj_b} ."
        inputs = self._det_processor(images=image, text=caption, return_tensors="pt").to(self.device)
        outputs = self._det_model(**inputs)
        results = self._det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        labels = results.get("text_labels", results.get("labels", []))
        cands_a: list[DetectionBox] = []
        cands_b: list[DetectionBox] = []

        for box_t, score_t, lbl in zip(results["boxes"], results["scores"], labels):
            x1, y1, x2, y2 = box_t.tolist()
            db = DetectionBox(
                label=str(lbl),
                score=float(score_t.item()),
                x1=max(0.0, min(1.0, x1 / w)),
                y1=max(0.0, min(1.0, y1 / h)),
                x2=max(0.0, min(1.0, x2 / w)),
                y2=max(0.0, min(1.0, y2 / h)),
            )
            if _match_label(db.label, obj_a):
                cands_a.append(db)
            if _match_label(db.label, obj_b):
                cands_b.append(db)

        cands_a.sort(key=lambda b: b.score, reverse=True)
        cands_b.sort(key=lambda b: b.score, reverse=True)

        best_a = cands_a[0] if cands_a else None
        best_b = None
        for cb in cands_b:
            if best_a is None or _box_iou(best_a, cb) < 0.85:
                best_b = cb
                break
        if best_b is None and cands_b:
            best_b = cands_b[0]
        return best_a, best_b

    @torch.no_grad()
    def encode_image_siglip(self, image: Image.Image) -> np.ndarray:
        self.load()
        assert self._sig_processor is not None and self._sig_model is not None
        inputs = self._sig_processor(images=image, return_tensors="pt").to(self.device)
        out = self._sig_model.get_image_features(**inputs)
        feats = out if torch.is_tensor(out) else out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats[0].detach().cpu().float().numpy()

    @torch.no_grad()
    def score_crop_colors(
        self,
        image: Image.Image,
        box: DetectionBox,
        obj_name: str,
        expected_color: str,
        contrast_color: str,
    ) -> float:
        self.load()
        assert self._sig_processor is not None and self._sig_model is not None
        w, h = image.size
        pad = 0.03
        px1 = max(0, int((box.x1 - pad) * w))
        py1 = max(0, int((box.y1 - pad) * h))
        px2 = min(w, max(px1 + 8, int((box.x2 + pad) * w)))
        py2 = min(h, max(py1 + 8, int((box.y2 + pad) * h)))
        crop = image.crop((px1, py1, px2, py2))

        texts = [f"a {expected_color} {obj_name}", f"a {contrast_color} {obj_name}"]
        inputs = self._sig_processor(
            text=texts, images=crop, padding="max_length", return_tensors="pt"
        ).to(self.device)
        outputs = self._sig_model(**inputs)
        # Use cosine similarity between normalized embeddings
        img_f = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        txt_f = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sims = (img_f @ txt_f.T)[0]
        return float((sims[0] - sims[1]).item())

    def evaluate_image(
        self,
        image: Image.Image,
        obj_a: str,
        obj_b: str,
        color_a: str,
        color_b: str,
        expected_relation: str = "none",
        compute_embedding: bool = True,
    ) -> ImageEvalResult:
        box_a, box_b = self.detect_two_objects(image, obj_a, obj_b)
        both = box_a is not None and box_b is not None

        dx: float | None = None
        rel_pred = "unknown"
        rel_correct: bool | None = None
        if both and box_a is not None and box_b is not None:
            dx = float(box_b.cx - box_a.cx)
            if dx > self.spatial_margin:
                rel_pred = "left"
            elif dx < -self.spatial_margin:
                rel_pred = "right"
            if expected_relation in ("left", "right"):
                rel_correct = rel_pred == expected_relation

        color_a_margin: float | None = None
        color_b_margin: float | None = None
        color_binding_correct: bool | None = None
        if box_a is not None:
            color_a_margin = self.score_crop_colors(image, box_a, obj_a, color_a, color_b)
        if box_b is not None:
            color_b_margin = self.score_crop_colors(image, box_b, obj_b, color_b, color_a)
        if color_a_margin is not None and color_b_margin is not None:
            color_binding_correct = (color_a_margin > 0.0) and (color_b_margin > 0.0)

        emb = self.encode_image_siglip(image) if compute_embedding else None
        return ImageEvalResult(
            box_a=box_a,
            box_b=box_b,
            both_detected=both,
            dx=dx,
            relation_pred=rel_pred,
            relation_correct=rel_correct,
            color_a_margin=color_a_margin,
            color_b_margin=color_b_margin,
            color_binding_correct=color_binding_correct,
            siglip_embedding=emb,
        )
