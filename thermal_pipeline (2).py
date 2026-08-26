"""
Phase 1: Data Acquisition & Preprocessing Pipeline
Tailored to real dataset structure observed from samples:

  DMR-IR / Kaggle : PAC_{id}_DN{num}-{dir|esq}.png
                    640×480 grayscale PNG, black background (pre-isolated)
  Mendeley        : IIR{id}_{anterior|oblleft|oblright}.jpg
                    ~240×300 JPEG, full torso, needs ROI extraction
"""

import os
import re
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ─────────────────────────────────────────────
# 0.  Config & Data Structures
# ─────────────────────────────────────────────

class ViewType(Enum):
    ANTERIOR      = "anterior"       # frontal
    LEFT_LATERAL  = "left_lateral"
    RIGHT_LATERAL = "right_lateral"
    LEFT_OBLIQUE  = "left_oblique"
    RIGHT_OBLIQUE = "right_oblique"
    UNKNOWN       = "unknown"

# Portuguese suffix → canonical view
_PAC_VIEW_MAP: Dict[str, ViewType] = {
    "esq": ViewType.LEFT_LATERAL,   # esquerda = left
    "dir": ViewType.RIGHT_LATERAL,  # direita  = right
}

# Mendeley suffix → canonical view
_IIR_VIEW_MAP: Dict[str, ViewType] = {
    "anterior" : ViewType.ANTERIOR,
    "oblleft"  : ViewType.LEFT_OBLIQUE,
    "oblright" : ViewType.RIGHT_OBLIQUE,
}


@dataclass
class ThermalSample:
    image:      np.ndarray          # float32  (H, W)
    patient_id: str
    dataset:    str                 # "DMR-IR" | "Kaggle" | "Mendeley"
    view:       ViewType = ViewType.UNKNOWN
    label:      Optional[int] = None    # 0=normal, 1=abnormal (if available)
    source_path: str = ""


@dataclass
class PipelineConfig:
    target_size:       Tuple[int, int] = (320, 240)  # (W, H) output
    clahe_clip:        float           = 2.0
    clahe_grid:        Tuple[int, int] = (8, 8)
    hot_pixel_thresh:  float           = 3.5          # std multiples
    roi_model_path:    Optional[str]   = None         # U-Net weights
    augment:           bool            = True
    aug_p:             float           = 0.5
    thermal_noise_std: float           = 0.02


# ─────────────────────────────────────────────
# Dataset Loaders  (exact naming from your files)
# ─────────────────────────────────────────────

# PAC_11_DN0-dir.png  →  patient=11, seq=0, view=dir
_PAC_RE = re.compile(r"PAC_(\d+)_DN(\d+)-(dir|esq)", re.IGNORECASE)

# IIR0019_anterior.jpg  →  patient=0019, view=anterior
_IIR_RE = re.compile(r"IIR(\d+)_(anterior|oblleft|oblright)", re.IGNORECASE)


def _read_gray_float(path: str) -> Optional[np.ndarray]:
    """Load any image as float32 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARNING] Could not read: {path}")
        return None
    return img.astype(np.float32)


# Folder-name vocabulary across all three datasets. Checked as two full passes
# (all healthy keys, then all patient keys) rather than interleaved, so that
# "Non Cancerous" is never mis-matched by the "CANCEROUS" substring inside it.
_HEALTHY_FOLDER_KEYS = ("SAUD", "NON CANCEROUS", "NONCANCEROUS", "NORMAL", "HEALTHY")
_PATIENT_FOLDER_KEYS = ("DOENTE", "CANCEROUS", "MALIGNANT", "BENIGN", "PATIENT")


def _detect_label_from_path(path: Path) -> Optional[int]:
    """
    Detect label from parent folder names across DMR-IR, Kaggle and Mendeley.

    Healthy/normal → 0 :  SAUDÁVEIS, "Non Cancerous", NORMAL, HEALTHY
    Patient/abnormal → 1 : DOENTES, "Cancerous", MALIGNANT, BENIGN, PATIENT

    Both Benign and Malignant map to 1 (patient/abnormal) — this file only
    resolves the coarse binary label; the malignant/benign distinction is
    handled downstream by the Metadata Agent where finer-grained diagnosis
    data (Diagnostics.xlsx) is available.
    """
    parts_upper = [part.upper() for part in path.parts]
    for part_upper in parts_upper:
        if any(k in part_upper for k in _HEALTHY_FOLDER_KEYS):
            return 0
    for part_upper in parts_upper:
        if any(k in part_upper for k in _PATIENT_FOLDER_KEYS):
            return 1
    return None   # label unknown


def load_pac_dataset(root: str, dataset_name: str = "DMR-IR") -> List[ThermalSample]:
    """
    Load PAC_XX_DNY-{dir|esq}.png files.
    Used for both DMR-IR and Kaggle (same naming convention).
    Automatically detects labels from SAUDÁVEIS / DOENTES subfolders.
    """
    samples = []
    skipped = 0
    for p in sorted(Path(root).rglob("*.png")):
        m = _PAC_RE.search(p.stem)
        if not m:
            skipped += 1
            continue
        patient_id, seq, view_suffix = m.groups()
        img = _read_gray_float(str(p))
        if img is None:
            skipped += 1
            continue
        label = _detect_label_from_path(p)
        samples.append(ThermalSample(
            image      = img,
            patient_id = f"PAC_{patient_id.zfill(2)}_DN{seq}",  # include seq to avoid collision
            dataset    = dataset_name,
            view       = _PAC_VIEW_MAP.get(view_suffix.lower(), ViewType.UNKNOWN),
            label      = label,
            source_path= str(p),
        ))
    label_counts = {0: sum(1 for s in samples if s.label == 0),
                    1: sum(1 for s in samples if s.label == 1),
                    None: sum(1 for s in samples if s.label is None)}
    print(f"[{dataset_name}] Loaded {len(samples)} images from {root}")
    print(f"  Healthy (0): {label_counts[0]}  |  Patient (1): {label_counts[1]}  |  Unknown: {label_counts[None]}")
    if skipped:
        print(f"  Skipped {skipped} non-matching files")
    return samples


def load_mendeley_dataset(root: str) -> List[ThermalSample]:
    """
    Load IIR{id}_{anterior|oblleft|oblright}.jpg files.
    Automatically detects labels from SAUDÁVEIS / DOENTES subfolders
    if present; otherwise label stays None.
    """
    samples = []
    skipped = 0
    for p in sorted(Path(root).rglob("*.jpg")):
        m = _IIR_RE.search(p.stem)
        if not m:
            skipped += 1
            continue
        patient_id, view_suffix = m.groups()
        img = _read_gray_float(str(p))
        if img is None:
            skipped += 1
            continue
        label = _detect_label_from_path(p)
        samples.append(ThermalSample(
            image      = img,
            patient_id = f"IIR{patient_id}",
            dataset    = "Mendeley",
            view       = _IIR_VIEW_MAP.get(view_suffix.lower(), ViewType.UNKNOWN),
            label      = label,
            source_path= str(p),
        ))
    label_counts = {0: sum(1 for s in samples if s.label == 0),
                    1: sum(1 for s in samples if s.label == 1),
                    None: sum(1 for s in samples if s.label is None)}
    print(f"[Mendeley] Loaded {len(samples)} images from {root}")
    print(f"  Healthy (0): {label_counts[0]}  |  Patient (1): {label_counts[1]}  |  Unknown: {label_counts[None]}")
    if skipped:
        print(f"  Skipped {skipped} non-matching files")
    return samples


# ─────────────────────────────────────────────
# STEP 1 — Thermal Normalization
# ─────────────────────────────────────────────

class ThermalNormalizer:
    """
    Min-max → z-score normalization.

    DMR-IR / Kaggle:  pixel values 0–255 grayscale (already windowed)
    Mendeley:         same 0–255, but higher contrast / different exposure
    Both are brought to zero-mean unit-variance before further steps.
    """

    @staticmethod
    def minmax(img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        if mx == mn:
            return np.zeros_like(img, dtype=np.float32)
        return (img - mn) / (mx - mn)

    @staticmethod
    def zscore(img: np.ndarray) -> np.ndarray:
        mu, sigma = img.mean(), img.std()
        if sigma == 0:
            return np.zeros_like(img, dtype=np.float32)
        return (img - mu) / sigma

    def __call__(self, sample: ThermalSample) -> ThermalSample:
        img = self.minmax(sample.image.astype(np.float32))
        img = self.zscore(img)
        sample.image = img
        return sample


# ─────────────────────────────────────────────
# STEP 2 — Cross-Dataset Harmonization
# ─────────────────────────────────────────────

class CrossDatasetHarmonizer:
    """
    Removes camera-specific artifacts and aligns thermal profiles.

    Observed differences:
    - DMR-IR / Kaggle : low-contrast grey, black bg already present
    - Mendeley        : higher contrast, strong white/bright thermal signal
    CLAHE equalizes these differences per-image.
    """

    def __init__(self, clip_limit: float = 2.0,
                 tile_grid: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid  = tile_grid

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        shifted = img - img.min()
        mx = shifted.max()
        if mx == 0:
            return np.zeros_like(shifted, dtype=np.uint8)
        return (shifted / mx * 255).astype(np.uint8)

    def remove_hot_pixels(self, img: np.ndarray,
                          thresh: float = 3.5) -> np.ndarray:
        """Replace spike pixels (camera sensor noise) with local median."""
        blurred = cv2.medianBlur(img.astype(np.float32), 5)
        diff    = np.abs(img - blurred)
        mask    = diff > thresh * diff.std()
        img[mask] = blurred[mask]
        return img

    def clahe(self, img: np.ndarray) -> np.ndarray:
        u8    = self._to_uint8(img)
        eq    = cv2.createCLAHE(self.clip_limit, self.tile_grid).apply(u8)
        return eq.astype(np.float32) / 255.0

    def __call__(self, sample: ThermalSample) -> ThermalSample:
        img = self.remove_hot_pixels(sample.image.copy())
        img = self.clahe(img)
        sample.image = img
        return sample


# ─────────────────────────────────────────────
# STEP 3 — Multi-View Alignment
# ─────────────────────────────────────────────

class MultiViewAligner:
    """
    Standardize all views to a common canvas size and orientation.

    Observed in your data:
    - PAC files:  breast ROI sits on black bg, LEFT-side in image 1
                  breast ROI sits on black bg, RIGHT-side in image 2
    - IIR files:  full torso, different JPEG resolution

    Canonical orientation: breast mass positioned on the LEFT side of frame
    (mirrors right-lateral views so all are spatially consistent).
    """

    # Views whose breast content should be mirrored to match left orientation
    _MIRROR_VIEWS = {ViewType.RIGHT_LATERAL, ViewType.RIGHT_OBLIQUE}

    def __init__(self, target_size: Tuple[int, int] = (320, 240)):
        self.target_size = target_size  # (W, H)

    def __call__(self, sample: ThermalSample) -> ThermalSample:
        img = cv2.resize(sample.image,
                         self.target_size,
                         interpolation=cv2.INTER_LINEAR)
        if sample.view in self._MIRROR_VIEWS:
            img = np.fliplr(img)
        sample.image = img
        return sample


# ─────────────────────────────────────────────
# STEP 4 — ROI Extraction
# ─────────────────────────────────────────────

class ROIExtractor:
    """
    Isolate the breast region and zero out everything else.

    Strategy differs by dataset:

    PAC (DMR-IR / Kaggle):
      The breast is ALREADY on a pure-black background.
      A simple non-zero mask is sufficient — no U-Net needed.

    IIR (Mendeley):
      Full torso is visible (arms, neck, background cloth).
      We use an adaptive threshold + largest-contour selection
      to keep only the two breast blobs.
      Replace with a U-Net for production use.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = self._load_unet(model_path)

    # ── U-Net stub (swap in your weights) ────────────────────────
    def _load_unet(self, path: str):
        try:
            import torch
            m = torch.load(path, map_location="cpu")
            m.eval()
            print(f"[ROIExtractor] U-Net loaded from {path}")
            return m
        except Exception as e:
            print(f"[ROIExtractor] Could not load U-Net: {e}")
            return None

    def _unet_mask(self, img: np.ndarray) -> np.ndarray:
        import torch
        x = torch.tensor(img[None, None]).float()
        with torch.no_grad():
            logits = self.model(x)
        return (torch.sigmoid(logits[0, 0]) > 0.5).numpy().astype(np.uint8)

    # ── PAC: already black background ────────────────────────────
    def _pac_mask(self, img_u8: np.ndarray) -> np.ndarray:
        """Keep any non-black pixel — PAC images already have black background."""
        mask = (img_u8 > 5).astype(np.uint8)
        # Fill small holes inside the breast region
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── Mendeley: flood-fill from border to isolate body ─────────
    def _mendeley_mask(self, img_u8: np.ndarray) -> np.ndarray:
        """
        Robust ROI for Mendeley full-torso JPEGs.

        ROOT CAUSE of previous failures:
        - Mendeley images have a MIXED background: top corners are BRIGHT
          (wall/clothing ~198 px) while bottom corners are DARK (0-9 px).
        - Simple brightness thresholds therefore either keep everything
          (threshold too low) or nothing (threshold too high).

        CORRECT APPROACH — Border Flood-Fill:
        1. Binarize: pixels < 25 are background (actual dark backdrop)
        2. Flood-fill starting from ALL border pixels that are dark
           → this grows outward through connected dark background
        3. Invert the filled region → body silhouette mask
        4. Keep the largest connected component (torso)
        5. Apply anatomical crop: remove top 12% (neck/head), trim sides

        Validated on IIR0001 & IIR0019 all views → 75-78% coverage,
        centroid at 56% from top (correct breast zone).
        """
        h, w = img_u8.shape

        # Step 1: Dark pixels = actual background (threshold at 25)
        _, body_rough = cv2.threshold(img_u8, 25, 255, cv2.THRESH_BINARY)
        body_inv = cv2.bitwise_not(body_rough)   # dark bg = 255

        # Step 2: Flood-fill from all border pixels that are dark
        flood_input = body_inv.copy()
        flood_mask  = np.zeros((h + 2, w + 2), np.uint8)

        border_pts = (
            [(0,    x) for x in range(0, w, 4)] +   # top edge
            [(h-1,  x) for x in range(0, w, 4)] +   # bottom edge
            [(y,    0) for y in range(0, h, 4)] +   # left edge
            [(y,  w-1) for y in range(0, h, 4)]    # right edge
        )
        for (r, c) in border_pts:
            if flood_input[r, c] == 255:   # dark border pixel
                cv2.floodFill(flood_input, flood_mask, (c, r), 128)

        # Pixels set to 128 = connected background; body = everything else
        bg_connected = (flood_input == 128).astype(np.uint8) * 255
        body_mask    = cv2.bitwise_not(bg_connected)

        # Step 3: Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, k_close)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN,  k_open)

        # Step 4: Keep only the largest blob (main torso)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask)
        final_mask = np.zeros((h, w), np.uint8)
        if n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            final_mask[labels == largest] = 255

        # Step 5: Anatomical crop — remove neck/head (top 12%), trim arms
        breast_roi = np.zeros((h, w), np.uint8)
        breast_roi[int(h * 0.12):, int(w * 0.05):int(w * 0.95)] = 255
        final_mask = cv2.bitwise_and(final_mask, breast_roi)

        # Final hole-fill
        k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k_fill)

        return (final_mask > 0).astype(np.uint8)

    def __call__(self, sample: ThermalSample) -> ThermalSample:
        img = sample.image   # float32 z-score from earlier steps

        # Rescale to 0-255 uint8 for all OpenCV threshold operations
        img_shifted = img - img.min()
        mx = float(img_shifted.max())
        img_u8 = (img_shifted / mx * 255).astype(np.uint8) if mx > 0 \
                 else np.zeros_like(img, dtype=np.uint8)

        if self.model is not None:
            mask = self._unet_mask(img_shifted / (mx + 1e-8))
        elif sample.dataset in ("DMR-IR", "Kaggle"):
            mask = self._pac_mask(img_u8)
        else:   # Mendeley
            mask = self._mendeley_mask(img_u8)

        sample.image = img * mask.astype(np.float32)
        return sample


# ─────────────────────────────────────────────
# STEP 5 — Data Augmentation
# ─────────────────────────────────────────────

class ThermalAugmentor:
    """
    Thermal-aware augmentations applied ONLY during training.

    Augmentations chosen to respect thermal physics:
    - Horizontal flip  : valid (bilateral symmetry assumption)
    - Small rotation   : ±10° (patient positioning variation)
    - Gaussian noise   : simulates sensor noise
    - Brightness shift : simulates ambient temperature drift
    - NO vertical flip : anatomically invalid for breast thermography
    """

    def __init__(self, p: float = 0.5, noise_std: float = 0.02):
        self.p         = p
        self.noise_std = noise_std

    def _maybe(self, condition: bool = True) -> bool:
        return condition and (np.random.rand() < self.p)

    def horizontal_flip(self, img: np.ndarray,
                        view: ViewType) -> Tuple[np.ndarray, ViewType]:
        """Flip image and update view label accordingly."""
        if not self._maybe():
            return img, view
        img = np.fliplr(img)
        flip_map = {
            ViewType.LEFT_LATERAL : ViewType.RIGHT_LATERAL,
            ViewType.RIGHT_LATERAL: ViewType.LEFT_LATERAL,
            ViewType.LEFT_OBLIQUE : ViewType.RIGHT_OBLIQUE,
            ViewType.RIGHT_OBLIQUE: ViewType.LEFT_OBLIQUE,
        }
        return img, flip_map.get(view, view)

    def rotate(self, img: np.ndarray,
               max_angle: float = 10.0) -> np.ndarray:
        if not self._maybe():
            return img
        angle = np.random.uniform(-max_angle, max_angle)
        h, w  = img.shape
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    def gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        if not self._maybe():
            return img
        std  = np.random.uniform(0, self.noise_std)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        # Zero noise on background (already masked to 0)
        noise[img == 0] = 0
        return img + noise

    def brightness_shift(self, img: np.ndarray,
                         max_delta: float = 0.05) -> np.ndarray:
        """Simulate ambient temperature drift (uniform offset on ROI)."""
        if not self._maybe():
            return img
        delta      = np.random.uniform(-max_delta, max_delta)
        shifted    = img.copy()
        shifted[img != 0] += delta       # only on breast pixels
        return shifted

    def synthetic_thermal_perturbation(self, img: np.ndarray) -> np.ndarray:
        """
        Smooth low-frequency noise simulating camera-to-camera thermal drift.
        Upsampled from a tiny random field → spatially correlated.
        """
        if not self._maybe(True):        # applied at lower probability
            return img
        h, w   = img.shape
        small  = np.random.randn(h // 8 or 1, w // 8 or 1).astype(np.float32)
        noise  = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        noise  = noise / (noise.std() + 1e-8) * (self.noise_std * 0.5)
        noise[img == 0] = 0
        return img + noise

    def __call__(self, sample: ThermalSample) -> ThermalSample:
        img, view = sample.image.copy(), sample.view

        img, view = self.horizontal_flip(img, view)
        img       = self.rotate(img)
        img       = self.gaussian_noise(img)
        img       = self.brightness_shift(img)
        img       = self.synthetic_thermal_perturbation(img)

        sample.image = img
        sample.view  = view
        return sample


# ─────────────────────────────────────────────
# Full Pipeline Orchestrator
# ─────────────────────────────────────────────

class ThermalPreprocessingPipeline:
    """Chains Steps 1–5 in order."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.cfg        = config
        self.normalizer = ThermalNormalizer()
        self.harmonizer = CrossDatasetHarmonizer(config.clahe_clip,
                                                 config.clahe_grid)
        self.aligner    = MultiViewAligner(config.target_size)
        self.extractor  = ROIExtractor(config.roi_model_path)
        self.augmentor  = (ThermalAugmentor(config.aug_p,
                                            config.thermal_noise_std)
                          if config.augment else None)

    def __call__(self, sample: ThermalSample,
                 augment: bool = False) -> ThermalSample:
        sample = self.normalizer(sample)     # Step 1
        sample = self.harmonizer(sample)     # Step 2
        sample = self.aligner(sample)        # Step 3
        sample = self.extractor(sample)      # Step 4
        if augment and self.augmentor:
            sample = self.augmentor(sample)  # Step 5
        return sample

    def process_batch(self, samples: List[ThermalSample],
                      augment: bool = False) -> List[ThermalSample]:
        return [self(s, augment=augment) for s in samples]


# ─────────────────────────────────────────────
# Utility: save processed images for inspection
# ─────────────────────────────────────────────

def save_processed(sample: ThermalSample, out_dir: str) -> Optional[str]:
    """
    Save a float32 processed image back to uint8 PNG.

    Folder structure created:
        out_dir/
            DMR-IR/
                healthy/   (label=0)
                patient/   (label=1)
                unknown/   (label=None)
            Mendeley/
                healthy/
                patient/
                unknown/
    Filename includes patient_id + view to guarantee uniqueness.
    Returns None (and prints a warning) if the image is blank after masking.
    """
    # Guard: skip blank images (failed ROI mask)
    if sample.image.max() == 0:
        print(f"  [SKIP] Blank image after masking: {sample.patient_id} {sample.view.value}")
        return None

    # Determine label subfolder
    label_dir = {0: "healthy", 1: "patient"}.get(sample.label, "unknown")

    # Full output directory: out_dir / dataset / label
    save_dir = os.path.join(out_dir, sample.dataset, label_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Rescale float32 → uint8
    img = sample.image
    mn, mx = img.min(), img.max()
    vis = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn           else np.zeros_like(img, dtype=np.uint8)

    # Unique filename: dataset_patientID_view.png
    # patient_id already contains DN seq (e.g. PAC_00_DN6) so no collision
    fname = f"{sample.patient_id}_{sample.view.value}.png"
    path  = os.path.join(save_dir, fname)
    cv2.imwrite(path, vis)
    return path


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_pipeline(
    dmr_ir_root:  Optional[str] = None,
    kaggle_root:  Optional[str] = None,
    mendeley_root: Optional[str] = None,
    output_dir:   str = "processed_output",
    augment_train: bool = True,
) -> List[ThermalSample]:
    """
    Load all three datasets, run the full pipeline, and save outputs.

    Args:
        dmr_ir_root   : folder with PAC_*.png  (DMR-IR)
        kaggle_root   : folder with PAC_*.png  (Kaggle Thermal)
        mendeley_root : folder with IIR*.jpg   (Mendeley)
        output_dir    : where to save processed PNGs
        augment_train : apply augmentation to the processed samples

    Returns:
        List of processed ThermalSample objects ready for model input.
    """
    raw: List[ThermalSample] = []

    if dmr_ir_root:
        raw += load_pac_dataset(dmr_ir_root,  dataset_name="DMR-IR")
    if kaggle_root:
        raw += load_pac_dataset(kaggle_root,  dataset_name="Kaggle")
    if mendeley_root:
        raw += load_mendeley_dataset(mendeley_root)

    print(f"\nTotal raw samples: {len(raw)}")

    config   = PipelineConfig(target_size=(320, 240), augment=augment_train)
    pipeline = ThermalPreprocessingPipeline(config)

    processed = []
    saved  = 0
    blanks = 0
    errors = 0

    for i, sample in enumerate(raw):
        try:
            # Always process without augmentation for saving
            # (augmentation is applied during training, not preprocessing)
            out  = pipeline(sample, augment=False)
            path = save_processed(out, output_dir)

            if path is None:
                blanks += 1
                print(f"  [{i+1}/{len(raw)}] BLANK  | {sample.dataset} | "
                      f"{sample.patient_id} | {sample.view.value}")
            else:
                saved += 1
                processed.append(out)
                print(f"  [{i+1}/{len(raw)}] SAVED  | {sample.dataset} | "
                      f"label={sample.label} | {sample.patient_id} | "
                      f"{sample.view.value} → {os.path.basename(path)}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(raw)}] ERROR  | {sample.source_path} | {e}")

    print(f"\n{'='*60}")
    print(f"Total input images  : {len(raw)}")
    print(f"Successfully saved  : {saved}")
    print(f"Blank (mask failed) : {blanks}")
    print(f"Errors              : {errors}")
    print(f"Output folder       : {os.path.abspath(output_dir)}")
    print(f"{'='*60}")
    return processed


# ─────────────────────────────────────────────
# Quick smoke-test with your actual filenames
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     import tempfile, shutil
#
#     print("=== Smoke Test: real filename patterns ===\n")
#
#     # Simulate PAC image (640×480, black background with a bright blob)
#     pac_img = np.zeros((480, 640), dtype=np.float32)
#     pac_img[100:400, 80:320] = np.random.uniform(100, 220,
#                                                   (300, 240)).astype(np.float32)
#
#     # Simulate IIR image (full torso, bright regions)
#     iir_img = np.random.uniform(30, 200, (300, 240)).astype(np.float32)
#     iir_img[60:240, 40:200] = np.random.uniform(160, 230, (180, 160))
#
#     samples = [
#         ThermalSample(pac_img.copy(), "PAC_11", "DMR-IR",
#                       ViewType.RIGHT_LATERAL, source_path="PAC_11_DN0-dir.png"),
#         ThermalSample(pac_img.copy(), "PAC_11", "DMR-IR",
#                       ViewType.LEFT_LATERAL,  source_path="PAC_11_DN0-esq.png"),
#         ThermalSample(pac_img.copy(), "PAC_00", "Kaggle",
#                       ViewType.RIGHT_LATERAL, source_path="PAC_00_DN6-dir.png"),
#         ThermalSample(iir_img.copy(), "IIR0019", "Mendeley",
#                       ViewType.ANTERIOR,      source_path="IIR0019_anterior.jpg"),
#         ThermalSample(iir_img.copy(), "IIR0019", "Mendeley",
#                       ViewType.LEFT_OBLIQUE,  source_path="IIR0019_oblleft.jpg"),
#         ThermalSample(iir_img.copy(), "IIR0019", "Mendeley",
#                       ViewType.RIGHT_OBLIQUE, source_path="IIR0019_oblright.jpg"),
#     ]
#
#     config   = PipelineConfig(target_size=(320, 240), augment=True)
#     pipeline = ThermalPreprocessingPipeline(config)
#
#     with tempfile.TemporaryDirectory() as tmp:
#         for s in samples:
#             out = pipeline(s, augment=True)
#             save_processed(out, tmp)
#             print(f"  ✓  {s.source_path:30s} | "
#                   f"out shape={out.image.shape} "
#                   f"min={out.image.min():.3f} max={out.image.max():.3f}")
#
#     print("\nAll pipeline steps completed successfully ✓")
if __name__ == "__main__":      # ← REPLACE everything inside this block

    results = run_pipeline(
        # dmr_ir_root   = "C:\PhD\Datasets\BC Mendely thermo\Breast Thermography",       # 📁 your PAC_*.png folder
        # kaggle_root   = "C:/PhD/Datasets/BC Kagggle thermo/breast-cancer-dataset/Train",        # 📁 your PAC_*.png folder
        mendeley_root = "C:\PhD\Datasets\BC Mendely thermo\Breast Thermography\Malignant",      # 📁 your IIR*.jpg folder
        output_dir    = "C:/datasets/processed-Mendely-Malignant",     # 📁 where to save output
        augment_train = True,
    )