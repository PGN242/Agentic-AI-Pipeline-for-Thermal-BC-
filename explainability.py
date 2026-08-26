"""
Phase 4: Explainability & Clinical Trust.

Four methods, adapted to the architecture actually built in this pipeline:

  Grad-CAM++ (Visual Attribution)
      Thermal heatmaps on the Imaging Agent's ResNet34 backbone — hooks
      `layer4`'s activations and gradients, computes the Grad-CAM++
      weighting (Chattopadhay et al., 2018) algebraically from a single
      backward pass, and overlays the result on the original thermogram.

  SHAP Values (Feature Importance)
      TreeExplainer on the Texture Agent's and Metadata Agent's
      GradientBoosting models — exact, fast, and appropriate for tree
      ensembles (no approximation needed, unlike KernelSHAP).

  Attention Visualization (the diagram's "Attention Rollout", adapted)
      The diagram's Attention Rollout traces layer-wise attention through a
      Vision Transformer. This pipeline's Imaging Agent is ResNet34 + a
      custom multi-view attention-pooling head (chosen for CPU training
      feasibility — see imaging_agent.py's docstring), which has no ViT
      layers to roll out through. What it DOES have is real, meaningful
      attention: a learned weight per view showing which of a subject's
      thermogram views the model leaned on most. That's what's visualized
      here — the same underlying question ("what did the model look at
      most"), answered with the attention this architecture actually has.

  Counterfactual XAI (Contrastive Reasoning)
      For the Texture Agent's tabular model: a greedy single-feature search
      over each feature's observed dataset range (holding all others fixed)
      for the smallest change that flips the prediction across its
      calibrated threshold — run through the deployed scale+select+GBM
      pipeline end-to-end, not a synthetic approximation of it.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib

from data_prep import SUBJECT_TABLE_CSV, subject_id_from_patient_id
from imaging_agent import ImagingAgent, image_to_tensor
from texture_classifier import (SUBJECT_FEATURES_CSV, ARTIFACT_PATH as TEXTURE_ARTIFACT_PATH)
from metadata_agent import MetadataAgent, FEATURE_NAMES as METADATA_FEATURE_NAMES, load_clinical_table

OUT_DIR = r"C:/PhD/Implementation/artifacts/explainability"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Grad-CAM++
# ─────────────────────────────────────────────

def _grad_cam_pp_map(A: torch.Tensor, G: torch.Tensor) -> np.ndarray:
    """A, G: [C, H, W] activation / gradient for one view. Returns an
    [H, W] map in [0, 1] (Chattopadhay et al., 2018, computed algebraically
    from first-order gradients — no higher-order autograd needed)."""
    grad2, grad3 = G ** 2, G ** 3
    sum_a_grad3 = (A * grad3).sum(dim=(1, 2), keepdim=True)
    alpha_denom = 2 * grad2 + sum_a_grad3
    alpha_denom = torch.where(alpha_denom.abs() > 1e-8, alpha_denom,
                               torch.full_like(alpha_denom, 1e-8))
    alpha = grad2 / alpha_denom
    weights = (alpha * F.relu(G)).sum(dim=(1, 2))          # [C]
    cam = F.relu((weights.view(-1, 1, 1) * A).sum(dim=0))  # [H, W]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def grad_cam_pp(imaging: ImagingAgent, subject_id: str, view_index: int = 0
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Returns (original_image[H,W] in [0,255], cam_map[H,W] in [0,1],
    predicted_probability), or None if the subject/view is unavailable."""
    if subject_id not in imaging.by_subject:
        return None
    views = imaging.by_subject[subject_id]
    if view_index >= len(views):
        return None

    tensors = [image_to_tensor(v.image, imaging.img_size, augment=False) for v in views]
    images = torch.stack(tensors, dim=0)
    images.requires_grad_(False)

    activations, gradients = {}, {}
    def fwd_hook(module, inp, out): activations["v"] = out
    def bwd_hook(module, gin, gout): gradients["v"] = gout[0]

    h1 = imaging.model.backbone.layer4.register_forward_hook(fwd_hook)
    h2 = imaging.model.backbone.layer4.register_full_backward_hook(bwd_hook)
    try:
        imaging.model.zero_grad()
        logits, attn = imaging.model(images, [len(tensors)])
        p = float(torch.softmax(logits, dim=1)[0, 1].item())
        target_class = int(p >= 0.5)
        logits[0, target_class].backward()

        A = activations["v"][view_index]   # [C, H, W]
        G = gradients["v"][view_index]
        cam = _grad_cam_pp_map(A, G)
    finally:
        h1.remove(); h2.remove()

    orig = views[view_index].image.astype(np.float32)
    mn, mx = orig.min(), orig.max()
    orig_u8 = ((orig - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    return orig_u8, cam, p


def grad_cam_overlay_figure(imaging: ImagingAgent, subject_id: str, view_index: int = 0,
                             true_label: Optional[int] = None):
    """Returns a matplotlib Figure (thermogram | Grad-CAM++ overlay), or None
    if the subject/view is unavailable. Used both to save PNGs to disk and
    to render live inside the Streamlit app (app.py)."""
    result = grad_cam_pp(imaging, subject_id, view_index)
    if result is None:
        return None
    orig, cam, p = result
    cam_resized = np.array(
        F.interpolate(torch.tensor(cam)[None, None], size=orig.shape,
                       mode="bilinear", align_corners=False)[0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axes[0].imshow(orig, cmap="gray"); axes[0].set_title("Thermogram", fontsize=10); axes[0].axis("off")
    axes[1].imshow(orig, cmap="gray")
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM++", fontsize=10); axes[1].axis("off")
    label_txt = f" | true={true_label}" if true_label is not None else ""
    fig.suptitle(f"{subject_id} — P(patient)={p:.2f}{label_txt}", fontsize=9)
    fig.tight_layout()
    return fig


def save_grad_cam_overlay(imaging: ImagingAgent, subject_id: str, out_path: str,
                           true_label: Optional[int] = None) -> Optional[str]:
    fig = grad_cam_overlay_figure(imaging, subject_id, true_label=true_label)
    if fig is None:
        return None
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_grad_cam_gallery(subject_ids: List[str], labels: Dict[str, int]) -> List[str]:
    imaging = ImagingAgent()
    paths = []
    for sid in subject_ids:
        safe = sid.replace("::", "_").replace("/", "_")
        out = f"{OUT_DIR}/gradcam_{safe}.png"
        p = save_grad_cam_overlay(imaging, sid, out, labels.get(sid))
        if p:
            paths.append(p)
            print(f"[GradCAM++] {sid} -> {p}")
    return paths


# ─────────────────────────────────────────────
# 2. SHAP values
# ─────────────────────────────────────────────

def _shap_bar(shap_values: np.ndarray, feature_names: List[str], title: str,
              out_path: str, top_k: int = 15) -> str:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = mean_abs[order][::-1]

    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.32 * len(names))))
    ax.barh(names, vals, color="#a32c62")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def texture_shap() -> Dict:
    import shap
    blob = joblib.load(TEXTURE_ARTIFACT_PATH)
    pipe, feat_cols = blob["pipeline"], blob["feature_names"]
    subj = pd.read_csv(SUBJECT_FEATURES_CSV)

    X_full = subj[feat_cols].to_numpy(dtype=np.float64)
    X_scaled = pipe.named_steps["scale"].transform(X_full)
    mask = pipe.named_steps["select"].get_support()
    X_sel = X_scaled[:, mask]
    selected_names = [f for f, m in zip(feat_cols, mask) if m]

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(X_sel)
    if isinstance(sv, list):   # some sklearn/shap versions return [class0, class1]
        sv = sv[1]

    out = _shap_bar(sv, selected_names, "Texture Agent — SHAP feature importance",
                     f"{OUT_DIR}/shap_texture_summary.png")
    return {"path": out, "top_feature": selected_names[int(np.argmax(np.abs(sv).mean(axis=0)))],
            "n_features": len(selected_names), "n_subjects": len(subj)}


def metadata_shap() -> Dict:
    import shap
    agent = MetadataAgent()
    df = load_clinical_table()
    X = df[METADATA_FEATURE_NAMES].to_numpy(dtype=np.float64)

    explainer = shap.TreeExplainer(agent.model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]

    out = _shap_bar(sv, METADATA_FEATURE_NAMES, "Metadata Agent — SHAP feature importance "
                     "(malignancy risk)", f"{OUT_DIR}/shap_metadata_summary.png")
    return {"path": out, "top_feature": METADATA_FEATURE_NAMES[int(np.argmax(np.abs(sv).mean(axis=0)))],
            "n_features": len(METADATA_FEATURE_NAMES), "n_subjects": len(df)}


def _texture_shap_row(subject_id: str):
    """Shared setup for a single subject's SHAP row — used by both the
    per-case explanation chart and the counterfactual search below."""
    import shap
    blob = joblib.load(TEXTURE_ARTIFACT_PATH)
    pipe, feat_cols = blob["pipeline"], blob["feature_names"]
    subj = pd.read_csv(SUBJECT_FEATURES_CSV).set_index("subject_id")
    if subject_id not in subj.index:
        return None

    X_full = subj[feat_cols].to_numpy(dtype=np.float64)
    X_scaled = pipe.named_steps["scale"].transform(X_full)
    mask = pipe.named_steps["select"].get_support()
    sel_names = [f for f, m in zip(feat_cols, mask) if m]
    idx = subj.index.get_loc(subject_id)

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv_row = explainer.shap_values(X_scaled[[idx]][:, mask])
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    return sel_names, sv_row[0], subj, pipe, feat_cols, mask


def texture_shap_for_subject(subject_id: str, top_k: int = 10) -> Optional[List[Tuple[str, float]]]:
    """Signed per-feature SHAP contributions for one subject, largest
    |value| first — positive pushes toward "patient", negative toward
    "healthy"."""
    result = _texture_shap_row(subject_id)
    if result is None:
        return None
    sel_names, sv_row, *_ = result
    order = np.argsort(np.abs(sv_row))[::-1][:top_k]
    return [(sel_names[i], float(sv_row[i])) for i in order]


def texture_shap_figure_for_subject(subject_id: str, top_k: int = 10):
    """Matplotlib horizontal bar chart of texture_shap_for_subject's output,
    for live rendering in the Streamlit app."""
    contributions = texture_shap_for_subject(subject_id, top_k)
    if not contributions:
        return None
    names = [c[0] for c in contributions][::-1]
    vals = [c[1] for c in contributions][::-1]
    colors = ["#a32c62" if v >= 0 else "#3d3579" for v in vals]

    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.34 * len(names))))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value  (+ toward patient, − toward healthy)")
    ax.set_title(f"{subject_id} — texture feature contributions", fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. Multi-view attention visualization (Attention Rollout analog)
# ─────────────────────────────────────────────

def attention_chart(subject_id: str, out_path: Optional[str] = None) -> Optional[str]:
    imaging = ImagingAgent()
    pred = imaging.predict(subject_id)
    if not pred.available or not pred.view_attention:
        return None
    views = imaging.by_subject[subject_id]
    if len(views) > imaging.max_views:
        idx = np.linspace(0, len(views) - 1, imaging.max_views).round().astype(int)
        views = [views[j] for j in idx]
    labels = [f"{v.view}\n#{i}" for i, v in enumerate(views)]

    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(labels)), 3.2))
    ax.bar(labels, pred.view_attention, color="#3d3579")
    ax.set_ylabel("attention weight")
    ax.set_title(f"{subject_id} — multi-view attention (P(patient)={pred.probability:.2f})",
                 fontsize=10)
    fig.tight_layout()
    out_path = out_path or f"{OUT_DIR}/attention_{subject_id.replace('::','_')}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────
# 4. Counterfactual explanation (Texture Agent, tabular)
# ─────────────────────────────────────────────

@dataclass
class Counterfactual:
    subject_id: str
    original_p: float
    found: bool
    feature: Optional[str] = None
    original_value: Optional[float] = None
    counterfactual_value: Optional[float] = None


def texture_counterfactual(subject_id: str, n_top_features: int = 8) -> Counterfactual:
    import shap
    blob = joblib.load(TEXTURE_ARTIFACT_PATH)
    pipe, feat_cols, threshold = blob["pipeline"], blob["feature_names"], blob["threshold"]
    subj = pd.read_csv(SUBJECT_FEATURES_CSV).set_index("subject_id")

    if subject_id not in subj.index:
        return Counterfactual(subject_id, 0.5, False)

    row = subj.loc[subject_id, feat_cols].to_numpy(dtype=np.float64)
    p0 = float(pipe.predict_proba(row.reshape(1, -1))[0, 1])
    target_side = -1 if p0 >= threshold else 1   # direction that would flip the call

    X_full = subj[feat_cols].to_numpy(dtype=np.float64)
    X_scaled = pipe.named_steps["scale"].transform(X_full)
    mask = pipe.named_steps["select"].get_support()
    sel_names = [f for f, m in zip(feat_cols, mask) if m]
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv_row = explainer.shap_values(X_scaled[[subj.index.get_loc(subject_id)]][:, mask])
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    ranked = [sel_names[i] for i in np.argsort(np.abs(sv_row[0]))[::-1][:n_top_features]]

    for fname in ranked:
        j = feat_cols.index(fname)
        col = X_full[:, j]
        lo, hi = np.percentile(col, [1, 99])
        candidates = np.linspace(lo, hi, 25) if target_side < 0 else np.linspace(hi, lo, 25)
        for cand in candidates:
            trial = row.copy()
            trial[j] = cand
            p_trial = float(pipe.predict_proba(trial.reshape(1, -1))[0, 1])
            flipped = (p_trial < threshold) if target_side < 0 else (p_trial >= threshold)
            if flipped:
                return Counterfactual(subject_id, p0, True, fname, float(row[j]), float(cand))
    return Counterfactual(subject_id, p0, False)


def run_counterfactual_batch(subject_ids: List[str]) -> pd.DataFrame:
    from dataclasses import asdict
    rows = [asdict(texture_counterfactual(sid)) for sid in subject_ids]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/counterfactuals.csv", index=False)
    return df


# ─────────────────────────────────────────────
# Entry point — run all four methods on a representative sample
# ─────────────────────────────────────────────

def run_all(sample_subject_ids: List[str], labels: Dict[str, int]) -> Dict:
    report: Dict = {}
    report["grad_cam_paths"] = run_grad_cam_gallery(sample_subject_ids, labels)
    report["texture_shap"] = texture_shap()
    report["metadata_shap"] = metadata_shap()
    report["attention_paths"] = [p for sid in sample_subject_ids
                                  if (p := attention_chart(sid)) is not None]
    report["counterfactuals"] = run_counterfactual_batch(sample_subject_ids).to_dict(orient="records")
    return report


if __name__ == "__main__":
    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    from data_prep import group_stratified_split
    _, _, test_ids = group_stratified_split(subjects_df, seed=42)
    label_map = subjects_df.set_index("subject_id")["label"].to_dict()
    sample = test_ids[:8]
    rep = run_all(sample, label_map)
    print(rep)
