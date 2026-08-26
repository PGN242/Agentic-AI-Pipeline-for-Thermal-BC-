"""
Phase 2: Texture Feature Agent for Breast Thermography

Consumes the output of the Phase-1 preprocessing pipeline (ROI-masked,
harmonized float32 images where background == 0) and produces a fixed-length,
named texture descriptor per image.

Four feature families:

  1. GLCM + LBP          -- second-order co-occurrence statistics (Haralick)
                            and multi-resolution local binary patterns
  2. Gabor filter bank   -- multi-scale / multi-orientation frequency response
  3. Thermal gradient    -- 1st/2nd order spatial derivatives, orientation
                            coherence, radial thermal drop-off, hot-spot metrics
  4. Statistical         -- first-order intensity descriptors + GLRLM
                            (gray-level run-length) texture statistics

Design notes specific to THERMAL data:
  - Background is exactly 0 after ROI masking. Every statistic is computed on
    ROI pixels only; convolution-based families (Gabor, gradient, LBP) first
    fill the background with the ROI mean so the ROI boundary does not create
    a phantom high-contrast edge, then sample inside an eroded mask.
  - Per-image min-max rescaling of ROI pixels makes descriptors invariant to
    camera gain / ambient offset (already partly handled by CLAHE upstream).
    Disable via TextureConfig.rescale_per_image if absolute levels matter.

Usage
-----
    from texture_agent import TextureAgent, TextureConfig

    agent = TextureAgent(TextureConfig())

    # (a) directly on Phase-1 samples (duck-typed: needs .image)
    feats = agent.extract_batch(processed_samples)

    # (b) on the PNGs written by save_processed()
    samples = load_processed_images("C:/datasets/processed-Mendely-Malignant")
    feats   = agent.extract_batch(samples)

    df = agent.to_dataframe(feats)
    df.to_csv("texture_features.csv", index=False)

The Phase-1 module lives in a file with spaces in its name, so import it with:

    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", "thermal_pipeline (2).py")
    tp   = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
"""

import os
import math
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Sequence

from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


_EPS = 1e-8


# ─────────────────────────────────────────────
# 0.  Config & Data Structures
# ─────────────────────────────────────────────

@dataclass
class TextureConfig:
    # ── global ────────────────────────────────────────────────
    rescale_per_image: bool = True      # min-max ROI pixels → [0, 1]
    mask_erode_px:     int   = 3        # erosion before filter-based sampling
    min_roi_pixels:    int   = 200      # below this the image is considered empty

    # ── family switches ───────────────────────────────────────
    use_glcm:        bool = True
    use_lbp:         bool = True
    use_gabor:       bool = True
    use_gradient:    bool = True
    use_statistical: bool = True

    # ── GLCM ──────────────────────────────────────────────────
    glcm_levels:    int              = 32
    glcm_distances: Tuple[int, ...]  = (1, 3, 5)
    glcm_angles:    Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    # ── LBP  (points, radius) pairs — multi-resolution ────────
    lbp_scales: Tuple[Tuple[int, int], ...] = ((8, 1), (16, 2), (24, 3))
    lbp_method: str = "uniform"        # rotation-invariant uniform patterns

    # ── Gabor bank ────────────────────────────────────────────
    # ~3–20 px structures at the Phase-1 output size of 320×240
    gabor_frequencies:  Tuple[float, ...] = (0.05, 0.1, 0.2, 0.35)        # cycles/px
    gabor_orientations: int   = 6                                        # 0..150°
    gabor_bandwidth:    float = 1.0                                      # octaves
    gabor_gamma:        float = 0.5                                      # aspect ratio

    # ── thermal gradient ──────────────────────────────────────
    grad_smooth_sigma:   float = 1.0
    grad_orient_bins:    int   = 8
    tensor_window:       int   = 7      # structure-tensor integration window
    hotspot_sigma:       float = 2.0    # hot-spot = mu + k*sigma of ROI
    radial_rings:        int   = 4      # rings for radial thermal profile

    # ── statistical / GLRLM ───────────────────────────────────
    hist_bins:     int = 64
    glrlm_levels:  int = 16


@dataclass
class TextureFeatures:
    """Named texture descriptor for a single thermal image."""
    features:    Dict[str, float]
    patient_id:  str  = ""
    dataset:     str  = ""
    view:        str  = ""
    label:       Optional[int] = None
    source_path: str  = ""
    valid:       bool = True            # False when the ROI was empty/degenerate

    @property
    def names(self) -> List[str]:
        return list(self.features.keys())

    def vector(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Feature vector in a fixed order (pass `names` to enforce an order)."""
        keys = list(names) if names is not None else self.names
        return np.array([self.features.get(k, np.nan) for k in keys],
                        dtype=np.float32)


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _finite(x: Any, fallback: float = 0.0) -> float:
    """Coerce to a finite python float (NaN/Inf → fallback)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink the ROI mask so boundary pixels do not pollute filter responses."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    er = cv2.erode(mask.astype(np.uint8), k)
    # If erosion wipes the ROI out (thin masks), keep the original.
    return er.astype(bool) if er.sum() > 0 else mask


def _fill_background(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background with the ROI mean → no artificial edge at the border."""
    out = img.copy()
    if mask.any():
        out[~mask] = float(img[mask].mean())
    return out


def _quantize(img: np.ndarray, mask: np.ndarray, levels: int,
              clip: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """
    Quantize ROI pixels to 1..levels; background stays 0.

    Percentile clipping keeps a single hot/cold outlier from collapsing the
    whole ROI into one bin — common in thermograms with a specular reflection.
    """
    q = np.zeros(img.shape, dtype=np.uint8)
    vals = img[mask]
    if vals.size == 0:
        return q
    lo, hi = np.percentile(vals, clip)
    if hi <= lo:
        q[mask] = 1
        return q
    v = np.clip(vals, lo, hi)
    q[mask] = 1 + np.floor((v - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)
    return q


# ─────────────────────────────────────────────
# FAMILY 1a — GLCM (Haralick) features
# ─────────────────────────────────────────────

class GLCMExtractor:
    """
    Gray-Level Co-occurrence Matrix descriptors.

    Background handling: the image is quantized so that background == symbol 0
    and ROI == symbols 1..L. The co-occurrence matrix is then computed over the
    whole frame and row/column 0 are zeroed out, which removes every pair that
    touches the background (including ROI↔background pairs at the boundary).

    Per distance we report the ANGLE MEAN (rotation-invariant summary) and the
    ANGLE RANGE (directionality — vascular / ductal structure is anisotropic).
    """

    _SKIMAGE_PROPS = ("contrast", "dissimilarity", "homogeneity",
                      "energy", "correlation", "ASM")

    def __init__(self, cfg: TextureConfig):
        self.cfg = cfg

    # ── manual props skimage does not provide ────────────────
    @staticmethod
    def _extra_props(P: np.ndarray) -> Dict[str, float]:
        """P: (L, L) normalized co-occurrence matrix over ROI symbols only."""
        L = P.shape[0]
        i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")

        mu_i = float((i * P).sum())
        mu_j = float((j * P).sum())
        sd_i = math.sqrt(float((((i - mu_i) ** 2) * P).sum()))
        sd_j = math.sqrt(float((((j - mu_j) ** 2) * P).sum()))

        nz = P[P > 0]
        entropy = float(-(nz * np.log2(nz)).sum())

        shade = float((((i + j - mu_i - mu_j) ** 3) * P).sum())
        prom  = float((((i + j - mu_i - mu_j) ** 4) * P).sum())
        # Normalize the 3rd/4th moments so they stay comparable across images
        shade /= (sd_i * sd_j + _EPS) ** 1.5
        prom  /= (sd_i * sd_j + _EPS) ** 2.0

        # Inverse difference moment normalized + variance
        idmn = float((P / (1.0 + ((i - j) ** 2) / (L ** 2))).sum())
        var  = float((((i - mu_i) ** 2) * P).sum())

        return {
            "entropy":            entropy,
            "cluster_shade":      shade,
            "cluster_prominence": prom,
            "max_probability":    float(P.max()),
            "idmn":               idmn,
            "variance":           var,
        }

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        cfg  = self.cfg
        L    = cfg.glcm_levels
        q    = _quantize(img, mask, L)
        dists  = list(cfg.glcm_distances)
        angles = list(cfg.glcm_angles)

        glcm = graycomatrix(q, distances=dists, angles=angles,
                            levels=L + 1, symmetric=True, normed=False)
        # Drop every pair involving the background symbol
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0

        out: Dict[str, float] = {}
        for pname in self._SKIMAGE_PROPS:
            vals = graycoprops(glcm, pname)              # (n_dist, n_angle)
            for di, d in enumerate(dists):
                row = np.asarray(vals[di], dtype=np.float64)
                row = row[np.isfinite(row)]
                if row.size == 0:
                    row = np.zeros(1)
                out[f"glcm_d{d}_{pname.lower()}_mean"]  = _finite(row.mean())
                out[f"glcm_d{d}_{pname.lower()}_range"] = _finite(row.max() - row.min())

        # Manual properties, computed per (distance, angle) then aggregated
        sub = glcm[1:, 1:, :, :].astype(np.float64)
        for di, d in enumerate(dists):
            per_angle: Dict[str, List[float]] = {}
            for ai in range(len(angles)):
                P = sub[:, :, di, ai]
                s = P.sum()
                if s <= 0:
                    continue
                for k, v in self._extra_props(P / s).items():
                    per_angle.setdefault(k, []).append(v)
            for k in ("entropy", "cluster_shade", "cluster_prominence",
                      "max_probability", "idmn", "variance"):
                arr = np.asarray(per_angle.get(k, [0.0]), dtype=np.float64)
                out[f"glcm_d{d}_{k}_mean"]  = _finite(arr.mean())
                out[f"glcm_d{d}_{k}_range"] = _finite(arr.max() - arr.min())
        return out


# ─────────────────────────────────────────────
# FAMILY 1b — Local Binary Patterns
# ─────────────────────────────────────────────

class LBPExtractor:
    """
    Multi-resolution rotation-invariant uniform LBP.

    For each (P, R) scale we keep the normalized P+2-bin histogram (sampled on
    the eroded mask so codes computed across the ROI border are discarded) plus
    three summary descriptors: histogram entropy, uniformity (energy) and the
    non-uniform-pattern fraction, which behaves as a micro-texture "roughness"
    index — elevated in the disorganized vascular patterns of malignant cases.
    """

    def __init__(self, cfg: TextureConfig):
        self.cfg = cfg

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        # LBP compares neighbours by strict inequality, so float inputs make the
        # codes sensitive to numerical dust — quantize to 8-bit first.
        filled = _fill_background(img, mask)
        mn, mx = float(filled.min()), float(filled.max())
        u8 = ((filled - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
             else np.zeros_like(filled, dtype=np.uint8)
        out: Dict[str, float] = {}

        for (P, R) in self.cfg.lbp_scales:
            codes = local_binary_pattern(u8, P, R, method=self.cfg.lbp_method)
            valid = _erode(mask, max(self.cfg.mask_erode_px, int(math.ceil(R))))
            vals  = codes[valid]

            n_bins = P + 2                              # uniform method → P+2 codes
            hist, _ = np.histogram(vals, bins=np.arange(n_bins + 1), density=False)
            hist = hist.astype(np.float64)
            total = hist.sum()
            hist = hist / total if total > 0 else hist

            tag = f"lbp_p{P}r{R}"
            for b in range(n_bins):
                out[f"{tag}_bin{b:02d}"] = _finite(hist[b])

            nz = hist[hist > 0]
            out[f"{tag}_entropy"]     = _finite(-(nz * np.log2(nz)).sum())
            out[f"{tag}_uniformity"]  = _finite((hist ** 2).sum())
            out[f"{tag}_nonuniform"]  = _finite(hist[-1])     # last bin = non-uniform
            out[f"{tag}_mean"]        = _finite(vals.mean() if vals.size else 0.0)
            out[f"{tag}_std"]         = _finite(vals.std()  if vals.size else 0.0)
        return out


# ─────────────────────────────────────────────
# FAMILY 2 — Gabor filter bank
# ─────────────────────────────────────────────

class GaborExtractor:
    """
    Even/odd Gabor quadrature pair per (frequency, orientation).

    Kernels are DC-free (mean subtracted) so a uniform temperature offset does
    not leak into the response. We store mean / std / energy of the response
    MAGNITUDE per filter, and per-frequency aggregates:

      *_orient_mean   : scale energy, independent of orientation
      *_orient_range  : anisotropy — how directional the texture is at that scale
      *_dom_orient    : dominant orientation (radians)
      *_orient_entropy: spread of energy across orientations (isotropy index)
    """

    def __init__(self, cfg: TextureConfig):
        self.cfg    = cfg
        self.thetas = [i * np.pi / cfg.gabor_orientations
                       for i in range(cfg.gabor_orientations)]
        self.bank   = self._build_bank()

    def _sigma_for(self, freq: float) -> float:
        """Sigma from frequency for a fixed octave bandwidth b."""
        b   = self.cfg.gabor_bandwidth
        lam = 1.0 / max(freq, _EPS)
        return (lam / np.pi) * math.sqrt(math.log(2) / 2) * \
               ((2 ** b + 1) / (2 ** b - 1))

    def _build_bank(self) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
        bank = []
        for f in self.cfg.gabor_frequencies:
            sigma = self._sigma_for(f)
            lam   = 1.0 / max(f, _EPS)
            ksz   = int(2 * math.ceil(3 * sigma) + 1)
            ksz   = max(7, min(ksz, 61))                  # keep kernels sane
            for th in self.thetas:
                kr = cv2.getGaborKernel((ksz, ksz), sigma, th, lam,
                                        self.cfg.gabor_gamma, 0.0, ktype=cv2.CV_32F)
                ki = cv2.getGaborKernel((ksz, ksz), sigma, th, lam,
                                        self.cfg.gabor_gamma, -np.pi / 2,
                                        ktype=cv2.CV_32F)
                kr = kr - kr.mean()                       # remove DC
                ki = ki - ki.mean()
                nrm = np.sqrt((kr ** 2).sum()) + _EPS     # unit L2 → comparable scales
                bank.append((f, th, kr / nrm, ki / nrm))
        return bank

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        filled = _fill_background(img, mask)
        valid  = _erode(mask, self.cfg.mask_erode_px)
        out: Dict[str, float] = {}

        # freq → list of (theta, mean energy) for the aggregates
        per_freq: Dict[float, List[Tuple[float, float]]] = {}

        for (f, th, kr, ki) in self.bank:
            re  = cv2.filter2D(filled, cv2.CV_32F, kr, borderType=cv2.BORDER_REFLECT)
            im  = cv2.filter2D(filled, cv2.CV_32F, ki, borderType=cv2.BORDER_REFLECT)
            mag = cv2.magnitude(re, im)
            v   = mag[valid]
            if v.size == 0:
                v = np.zeros(1, dtype=np.float32)

            tag = f"gabor_f{f:g}_t{int(round(np.degrees(th))):03d}"
            out[f"{tag}_mean"]   = _finite(v.mean())
            out[f"{tag}_std"]    = _finite(v.std())
            out[f"{tag}_energy"] = _finite((v ** 2).mean())
            per_freq.setdefault(f, []).append((th, float(v.mean())))

        for f, pairs in per_freq.items():
            thetas = np.array([p[0] for p in pairs])
            e      = np.array([p[1] for p in pairs], dtype=np.float64)
            tag    = f"gabor_f{f:g}"
            out[f"{tag}_orient_mean"]  = _finite(e.mean())
            out[f"{tag}_orient_std"]   = _finite(e.std())
            out[f"{tag}_orient_range"] = _finite(e.max() - e.min())
            out[f"{tag}_dom_orient"]   = _finite(thetas[int(np.argmax(e))])

            p = e / (e.sum() + _EPS)
            nz = p[p > 0]
            out[f"{tag}_orient_entropy"] = _finite(-(nz * np.log2(nz)).sum())
            # Anisotropy on doubled angles (orientation is π-periodic)
            r = np.abs((p * np.exp(2j * thetas)).sum())
            out[f"{tag}_anisotropy"] = _finite(r)

        # Cross-scale summary: where does the energy sit in the scale pyramid?
        freqs   = np.array(sorted(per_freq.keys()), dtype=np.float64)
        scale_e = np.array([np.mean([p[1] for p in per_freq[f]]) for f in freqs])
        w = scale_e / (scale_e.sum() + _EPS)
        out["gabor_scale_centroid"] = _finite((w * freqs).sum())
        nz = w[w > 0]
        out["gabor_scale_entropy"]  = _finite(-(nz * np.log2(nz)).sum())
        out["gabor_total_energy"]   = _finite(scale_e.sum())
        return out


# ─────────────────────────────────────────────
# FAMILY 3 — Thermal gradient analysis
# ─────────────────────────────────────────────

class ThermalGradientExtractor:
    """
    Spatial derivative descriptors with a thermographic reading:

      grad_mag_*        : steepness of thermal transitions (vascular edges)
      grad_orient_*     : orientation histogram entropy / circular anisotropy —
                          organized vasculature is directional, chaotic
                          neovascularization is not
      grad_coherence_*  : structure-tensor coherence, same idea, local
      lap_*             : 2nd derivative — thermal curvature / focal peaks
      radial_*          : temperature vs distance from ROI centroid; malignant
                          patterns often show a sharper central hot core
      hotspot_*         : suprathreshold (mu + k*sigma) region count/area and
                          the gradient magnitude on their boundaries
      ridge_density     : fraction of ROI above the 90th-percentile gradient
    """

    def __init__(self, cfg: TextureConfig):
        self.cfg = cfg

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        cfg    = self.cfg
        filled = _fill_background(img, mask)
        sm     = cv2.GaussianBlur(filled, (0, 0), cfg.grad_smooth_sigma)
        valid  = _erode(mask, cfg.mask_erode_px)

        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = np.arctan2(gy, gx)                      # (-π, π]

        out: Dict[str, float] = {}
        m = mag[valid]
        if m.size == 0:
            m = np.zeros(1, dtype=np.float32)

        out["grad_mag_mean"]   = _finite(m.mean())
        out["grad_mag_std"]    = _finite(m.std())
        out["grad_mag_max"]    = _finite(m.max())
        out["grad_mag_p90"]    = _finite(np.percentile(m, 90))
        out["grad_mag_energy"] = _finite((m ** 2).mean())
        out["grad_mag_skew"]   = _finite(sp_stats.skew(m)     if m.size > 2 else 0.0)
        out["ridge_density"]   = _finite(float((m > np.percentile(m, 90)).mean()))

        # ── orientation statistics (π-periodic → doubled angles) ──
        a = ang[valid]
        w = mag[valid].astype(np.float64)
        wsum = w.sum() + _EPS
        a2 = (2 * a) % (2 * np.pi)
        hist, _ = np.histogram(a2, bins=cfg.grad_orient_bins,
                               range=(0, 2 * np.pi), weights=w)
        p  = hist / (hist.sum() + _EPS)
        nz = p[p > 0]
        out["grad_orient_entropy"] = _finite(-(nz * np.log2(nz)).sum())
        R = np.abs((w * np.exp(1j * a2)).sum()) / wsum
        out["grad_orient_anisotropy"] = _finite(R)               # 0 = isotropic
        out["grad_orient_circvar"]    = _finite(1.0 - R)
        out["grad_dom_orient"]        = _finite(0.5 * float(
            np.angle((w * np.exp(1j * a2)).sum())))

        # ── structure-tensor coherence ────────────────────────────
        win = (cfg.tensor_window, cfg.tensor_window)
        Jxx = cv2.boxFilter(gx * gx, cv2.CV_32F, win)
        Jyy = cv2.boxFilter(gy * gy, cv2.CV_32F, win)
        Jxy = cv2.boxFilter(gx * gy, cv2.CV_32F, win)
        num = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
        coh = num / (Jxx + Jyy + _EPS)
        c = coh[valid]
        out["grad_coherence_mean"] = _finite(c.mean() if c.size else 0.0)
        out["grad_coherence_std"]  = _finite(c.std()  if c.size else 0.0)

        # ── 2nd order: thermal curvature ──────────────────────────
        lap = cv2.Laplacian(sm, cv2.CV_32F, ksize=3)
        l = np.abs(lap[valid])
        if l.size == 0:
            l = np.zeros(1, dtype=np.float32)
        out["lap_abs_mean"] = _finite(l.mean())
        out["lap_abs_std"]  = _finite(l.std())
        out["lap_abs_p95"]  = _finite(np.percentile(l, 95))

        # ── radial thermal profile from the ROI centroid ──────────
        out.update(self._radial_profile(img, mask))

        # ── hot-spot morphology ───────────────────────────────────
        out.update(self._hotspots(img, mask, mag))
        return out

    def _radial_profile(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Linear fit of temperature against normalized distance to centroid."""
        cfg = self.cfg
        ys, xs = np.nonzero(mask)
        out: Dict[str, float] = {}
        if ys.size < cfg.min_roi_pixels:
            out["radial_slope"] = 0.0
            out["radial_corr"]  = 0.0
            for r in range(cfg.radial_rings):
                out[f"radial_ring{r}_mean"] = 0.0
            out["radial_core_edge_ratio"] = 0.0
            return out

        cy, cx = ys.mean(), xs.mean()
        d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        d = d / (d.max() + _EPS)
        t = img[mask].astype(np.float64)

        slope, _, r_val, _, _ = sp_stats.linregress(d, t)
        out["radial_slope"] = _finite(slope)
        out["radial_corr"]  = _finite(r_val)

        edges = np.linspace(0, 1, cfg.radial_rings + 1)
        means = []
        for r in range(cfg.radial_rings):
            sel = (d >= edges[r]) & (d <= edges[r + 1])
            mval = float(t[sel].mean()) if sel.any() else 0.0
            out[f"radial_ring{r}_mean"] = _finite(mval)
            means.append(mval)
        out["radial_core_edge_ratio"] = _finite(means[0] / (means[-1] + _EPS))
        return out

    def _hotspots(self, img: np.ndarray, mask: np.ndarray,
                  mag: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        out: Dict[str, float] = {
            "hotspot_area_frac": 0.0, "hotspot_count": 0.0,
            "hotspot_max_area_frac": 0.0, "hotspot_border_grad": 0.0,
            "hotspot_contrast": 0.0,
        }
        vals = img[mask]
        if vals.size < cfg.min_roi_pixels:
            return out

        thr = vals.mean() + cfg.hotspot_sigma * vals.std()
        hot = ((img >= thr) & mask).astype(np.uint8)
        if hot.sum() == 0:
            return out

        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, k)
        roi_area = float(mask.sum())
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(hot)

        out["hotspot_area_frac"] = _finite(hot.sum() / roi_area)
        out["hotspot_count"]     = float(max(n_lab - 1, 0))
        if n_lab > 1:
            out["hotspot_max_area_frac"] = _finite(
                float(stats[1:, cv2.CC_STAT_AREA].max()) / roi_area)

        border = cv2.morphologyEx(hot, cv2.MORPH_GRADIENT, k).astype(bool) & mask
        if border.any():
            out["hotspot_border_grad"] = _finite(mag[border].mean())
        hb = hot.astype(bool)
        if hb.any():
            cold = mask & ~hb
            if cold.any():
                out["hotspot_contrast"] = _finite(
                    float(img[hb].mean() - img[cold].mean()))
        return out


# ─────────────────────────────────────────────
# FAMILY 4 — Statistical descriptors (+ GLRLM)
# ─────────────────────────────────────────────

class StatisticalExtractor:
    """
    First-order intensity statistics over ROI pixels, plus GLRLM (gray-level
    run-length) descriptors — the classical statistical texture family that
    captures how far uniform-temperature runs extend before the level changes.
    GLRLM features are averaged over the four principal directions.
    """

    _GLRLM_KEYS = ("sre", "lre", "gln", "rln", "rp",
                   "lgre", "hgre", "srlge", "srhge", "lrlge", "lrhge")

    def __init__(self, cfg: TextureConfig):
        self.cfg = cfg

    # ── first order ──────────────────────────────────────────
    def _first_order(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        v = img[mask].astype(np.float64)
        out: Dict[str, float] = {}
        if v.size == 0:
            v = np.zeros(1)

        mean = v.mean()
        std  = v.std()
        out["stat_mean"]     = _finite(mean)
        out["stat_std"]      = _finite(std)
        out["stat_var"]      = _finite(v.var())
        out["stat_skew"]     = _finite(sp_stats.skew(v)     if v.size > 2 else 0.0)
        out["stat_kurtosis"] = _finite(sp_stats.kurtosis(v) if v.size > 3 else 0.0)
        # Under per-image rescaling min/max/range are constants (0/1/1) and
        # would enter the model as dead columns — only emit them when the
        # absolute level is preserved.
        if not self.cfg.rescale_per_image:
            out["stat_min"]   = _finite(v.min())
            out["stat_max"]   = _finite(v.max())
            out["stat_range"] = _finite(v.max() - v.min())
        out["stat_median"]   = _finite(np.median(v))
        out["stat_rms"]      = _finite(np.sqrt((v ** 2).mean()))
        out["stat_mad"]      = _finite(np.mean(np.abs(v - mean)))
        out["stat_robust_mad"] = _finite(np.median(np.abs(v - np.median(v))))
        out["stat_cv"]       = _finite(std / (abs(mean) + _EPS))

        for p in (5, 10, 25, 75, 90, 95, 99):
            out[f"stat_p{p}"] = _finite(np.percentile(v, p))
        out["stat_iqr"] = _finite(np.percentile(v, 75) - np.percentile(v, 25))
        # Upper-tail mass: how much of the ROI sits in the hottest decile band
        out["stat_hot_tail"] = _finite(float((v >= np.percentile(v, 90)).mean()))

        hist, _ = np.histogram(v, bins=self.cfg.hist_bins)
        p = hist.astype(np.float64) / (hist.sum() + _EPS)
        nz = p[p > 0]
        out["stat_entropy"]    = _finite(-(nz * np.log2(nz)).sum())
        out["stat_uniformity"] = _finite((p ** 2).sum())
        out["stat_energy"]     = _finite((v ** 2).sum() / v.size)
        return out

    # ── GLRLM ────────────────────────────────────────────────
    @staticmethod
    def _runs(line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run-length encode a 1D array, dropping background (0) runs."""
        n = line.size
        if n == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        cuts   = np.flatnonzero(np.diff(line)) + 1
        starts = np.concatenate(([0], cuts))
        ends   = np.concatenate((cuts, [n]))
        vals   = line[starts].astype(np.int64)
        lens   = (ends - starts).astype(np.int64)
        keep   = vals > 0
        return vals[keep], lens[keep]

    @staticmethod
    def _lines(q: np.ndarray, direction: str) -> List[np.ndarray]:
        h, w = q.shape
        if direction == "0":                       # horizontal
            return [q[i, :] for i in range(h)]
        if direction == "90":                      # vertical
            return [q[:, j] for j in range(w)]
        if direction == "135":                     # main diagonals ↘
            return [np.diagonal(q, offset=o) for o in range(-h + 1, w)]
        f = np.fliplr(q)                           # anti-diagonals ↙  (45°)
        return [np.diagonal(f, offset=o) for o in range(-h + 1, w)]

    def _glrlm_one(self, q: np.ndarray, direction: str,
                   levels: int, n_roi: int) -> Dict[str, float]:
        max_run = max(q.shape)
        P = np.zeros((levels, max_run), dtype=np.float64)
        for line in self._lines(q, direction):
            vals, lens = self._runs(np.asarray(line))
            if vals.size == 0:
                continue
            np.add.at(P, (vals - 1, np.minimum(lens, max_run) - 1), 1.0)

        Nr = P.sum()
        if Nr <= 0:
            return {k: 0.0 for k in self._GLRLM_KEYS}

        i = np.arange(1, levels + 1, dtype=np.float64)[:, None]     # gray level
        j = np.arange(1, max_run + 1, dtype=np.float64)[None, :]    # run length

        return {
            "sre":   float((P / j ** 2).sum() / Nr),
            "lre":   float((P * j ** 2).sum() / Nr),
            "gln":   float((P.sum(axis=1) ** 2).sum() / Nr),
            "rln":   float((P.sum(axis=0) ** 2).sum() / Nr),
            "rp":    float(Nr / max(n_roi, 1)),
            "lgre":  float((P / i ** 2).sum() / Nr),
            "hgre":  float((P * i ** 2).sum() / Nr),
            "srlge": float((P / (i ** 2 * j ** 2)).sum() / Nr),
            "srhge": float((P * i ** 2 / j ** 2).sum() / Nr),
            "lrlge": float((P * j ** 2 / i ** 2).sum() / Nr),
            "lrhge": float((P * i ** 2 * j ** 2).sum() / Nr),
        }

    def _glrlm(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        levels = self.cfg.glrlm_levels
        q = _quantize(img, mask, levels)
        n_roi = int(mask.sum())

        per_dir = [self._glrlm_one(q, d, levels, n_roi)
                   for d in ("0", "45", "90", "135")]
        out: Dict[str, float] = {}
        for k in self._GLRLM_KEYS:
            arr = np.array([d[k] for d in per_dir], dtype=np.float64)
            out[f"glrlm_{k}_mean"]  = _finite(arr.mean())
            out[f"glrlm_{k}_range"] = _finite(arr.max() - arr.min())
        return out

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        out = self._first_order(img, mask)
        out.update(self._glrlm(img, mask))
        # ROI geometry — context for every intensity statistic above
        h, w = img.shape
        out["stat_roi_frac"] = _finite(float(mask.sum()) / (h * w))
        return out


# ─────────────────────────────────────────────
# The Agent
# ─────────────────────────────────────────────

class TextureAgent:
    """
    Orchestrates the four texture families into one named descriptor.

    Accepts either a raw np.ndarray or any Phase-1 `ThermalSample`-like object
    exposing `.image` (plus optional .patient_id/.dataset/.view/.label).
    """

    def __init__(self, config: TextureConfig = TextureConfig()):
        self.cfg  = config
        self.glcm = GLCMExtractor(config)              if config.use_glcm     else None
        self.lbp  = LBPExtractor(config)               if config.use_lbp      else None
        self.gabor = GaborExtractor(config)            if config.use_gabor    else None
        self.grad = ThermalGradientExtractor(config)   if config.use_gradient else None
        self.stat = StatisticalExtractor(config)       if config.use_statistical else None
        self._reference_names: Optional[List[str]] = None

    # ── input handling ───────────────────────────────────────
    @staticmethod
    def _as_image(obj: Any) -> np.ndarray:
        img = obj if isinstance(obj, np.ndarray) else getattr(obj, "image", None)
        if img is None:
            raise TypeError("Input must be an ndarray or expose an `.image` attribute")
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return img

    def _prepare(self, img: np.ndarray,
                 mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Derive the ROI mask and rescale ROI pixels to [0, 1]."""
        if mask is None:
            mask = np.abs(img) > _EPS          # Phase-1 zeroes the background
        mask = mask.astype(bool)
        # An all-zero mask means the image itself is blank (ROI extraction
        # failed upstream) — leave it empty so extract() flags the row invalid.

        out = img.astype(np.float32).copy()
        if self.cfg.rescale_per_image and mask.any():
            v = out[mask]
            mn, mx = float(v.min()), float(v.max())
            out = (out - mn) / (mx - mn) if mx > mn else np.zeros_like(out)
        out[~mask] = 0.0
        return out, mask

    # ── main API ─────────────────────────────────────────────
    def extract(self, sample: Any,
                mask: Optional[np.ndarray] = None) -> TextureFeatures:
        img = self._as_image(sample)
        img, roi = self._prepare(img, mask)

        view = getattr(sample, "view", "")
        view = getattr(view, "value", view)         # ViewType enum → str

        meta = dict(
            patient_id  = str(getattr(sample, "patient_id", "")),
            dataset     = str(getattr(sample, "dataset", "")),
            view        = str(view),
            label       = getattr(sample, "label", None),
            source_path = str(getattr(sample, "source_path", "")),
        )

        if int(roi.sum()) < self.cfg.min_roi_pixels:
            # Degenerate ROI (blank image / failed upstream mask) — emit the
            # canonical schema filled with NaN so the feature matrix stays
            # rectangular, and flag the row as invalid.
            return TextureFeatures(features={k: float("nan") for k in self.schema()},
                                   valid=False, **meta)

        feats = self._run_families(img, roi)
        if self._reference_names is None:
            self._reference_names = list(feats.keys())
        return TextureFeatures(features=feats, valid=True, **meta)

    def schema(self) -> List[str]:
        """Canonical feature names/order for the current config (cached)."""
        if self._reference_names is None:
            rng   = np.random.RandomState(0)
            dummy = rng.rand(96, 96).astype(np.float32)
            self._reference_names = list(
                self._run_families(dummy, np.ones((96, 96), dtype=bool)).keys())
        return self._reference_names

    def _run_families(self, img: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for extractor in (self.glcm, self.lbp, self.gabor, self.grad, self.stat):
            if extractor is None:
                continue
            feats.update(extractor(img, roi))
        return feats

    def extract_batch(self, samples: Sequence[Any],
                      verbose: bool = True) -> List[TextureFeatures]:
        out: List[TextureFeatures] = []
        n = len(samples)
        for i, s in enumerate(samples):
            try:
                f = self.extract(s)
                out.append(f)
                if verbose:
                    tag = "OK   " if f.valid else "EMPTY"
                    print(f"  [{i+1}/{n}] {tag} | {f.dataset} | label={f.label} | "
                          f"{f.patient_id} | {f.view} | {len(f.features)} features")
            except Exception as e:
                if verbose:
                    src = getattr(s, "source_path", "<array>")
                    print(f"  [{i+1}/{n}] ERROR | {src} | {e}")
        return out

    # ── export ───────────────────────────────────────────────
    @staticmethod
    def to_dataframe(features: Sequence[TextureFeatures]):
        """Metadata columns first, then the feature columns in a stable order."""
        import pandas as pd
        rows = []
        for f in features:
            row = {"patient_id": f.patient_id, "dataset": f.dataset,
                   "view": f.view, "label": f.label,
                   "source_path": f.source_path, "valid": f.valid}
            row.update(f.features)
            rows.append(row)
        return pd.DataFrame(rows)

    def feature_matrix(self, features: Sequence[TextureFeatures]
                       ) -> Tuple[np.ndarray, List[str]]:
        """(N, D) float32 matrix + the column names, in a fixed order."""
        names = self.schema()
        X = np.stack([f.vector(names) for f in features]) if features \
            else np.zeros((0, len(names)), np.float32)
        return X.astype(np.float32), list(names)


# ─────────────────────────────────────────────
# Loader for Phase-1 output PNGs
# ─────────────────────────────────────────────

@dataclass
class _LoadedSample:
    """Minimal ThermalSample stand-in so the agent can run standalone."""
    image:       np.ndarray
    patient_id:  str = ""
    dataset:     str = ""
    view:        str = ""
    label:       Optional[int] = None
    source_path: str = ""


_LABEL_DIRS = {"healthy": 0, "patient": 1, "unknown": None}

# Phase-1 filenames are "<patient_id>_<view>.png" where <view> is a ViewType
# value — and those contain underscores, so match the suffix explicitly.
_VIEW_NAMES = ("anterior", "left_lateral", "right_lateral",
               "left_oblique", "right_oblique", "unknown")


def _split_stem(stem: str) -> Tuple[str, str]:
    for v in _VIEW_NAMES:
        if stem.endswith("_" + v):
            return stem[: -(len(v) + 1)], v
    pid, _, view = stem.rpartition("_")
    return (pid or stem), view


def load_processed_images(root: str) -> List[_LoadedSample]:
    """
    Read the folder tree written by Phase-1 `save_processed()`:

        root/<dataset>/<healthy|patient|unknown>/<patient_id>_<view>.png

    Falls back gracefully when the tree is flatter than expected.
    """
    samples: List[_LoadedSample] = []
    for p in sorted(Path(root).rglob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARNING] Could not read: {p}")
            continue

        parts = [x.lower() for x in p.parts]
        label = next((_LABEL_DIRS[x] for x in parts if x in _LABEL_DIRS), None)
        dataset = p.parent.parent.name if p.parent.name.lower() in _LABEL_DIRS \
                  else p.parent.name

        pid, view = _split_stem(p.stem)
        samples.append(_LoadedSample(
            image       = img.astype(np.float32),
            patient_id  = pid,
            dataset     = dataset,
            view        = view,
            label       = label,
            source_path = str(p),
        ))

    counts = {0: sum(1 for s in samples if s.label == 0),
              1: sum(1 for s in samples if s.label == 1),
              None: sum(1 for s in samples if s.label is None)}
    print(f"[TextureAgent] Loaded {len(samples)} processed images from {root}")
    print(f"  Healthy (0): {counts[0]}  |  Patient (1): {counts[1]}  |  "
          f"Unknown: {counts[None]}")
    return samples


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_texture_extraction(
    processed_dir: str,
    output_csv: str = "texture_features.csv",
    config: Optional[TextureConfig] = None,
):
    """
    Load Phase-1 output, extract all texture families, write a CSV.

    Returns (DataFrame, List[TextureFeatures]).
    """
    cfg     = config or TextureConfig()
    agent   = TextureAgent(cfg)
    samples = load_processed_images(processed_dir)
    if not samples:
        print("[TextureAgent] Nothing to process.")
        return None, []

    print(f"\nExtracting texture features from {len(samples)} images...")
    feats = agent.extract_batch(samples)

    df = agent.to_dataframe(feats)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    X, names = agent.feature_matrix(feats)
    n_valid = int(sum(1 for f in feats if f.valid))

    print(f"\n{'='*60}")
    print(f"Images processed    : {len(feats)}")
    print(f"Valid descriptors   : {n_valid}")
    print(f"Feature dimension   : {X.shape[1]}")
    for fam, pre in (("GLCM", "glcm_"), ("LBP", "lbp_"), ("Gabor", "gabor_"),
                     ("Gradient", ("grad_", "lap_", "radial_", "hotspot_", "ridge_")),
                     ("Statistical", ("stat_", "glrlm_"))):
        pres = (pre,) if isinstance(pre, str) else pre
        print(f"  {fam:<12}: {sum(1 for n in names if n.startswith(pres))}")
    print(f"Output CSV          : {os.path.abspath(output_csv)}")
    print(f"{'='*60}")
    return df, feats


if __name__ == "__main__":

    df, feats = run_texture_extraction(
        processed_dir = "C:/PhD\Datasets/BC Kagggle thermo/breast-cancer-dataset",   # 📁 Phase-1 output
        output_csv    = "C:/datasets/texture_features_kaggle.csv", # 📁 feature table
        config        = TextureConfig(),
    )
