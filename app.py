"""
Interactive console for the Thermal Triage Pipeline.

Two modes:
  Browse dataset     — pick any of the 231 processed subjects and see the
                        full pipeline run live: Orchestrator routing,
                        per-agent outputs, uncertainty quantification, and
                        all four explainability methods.
  Upload thermogram   — quick single-image check through the Imaging Agent
                        only (clearly labeled as simplified: it skips the
                        dataset-specific ROI harmonization in
                        thermal_pipeline (2).py and the Texture/Metadata/
                        Orchestrator agents, which need full per-subject
                        multi-view context that a single upload can't
                        supply).

Run with:  streamlit run app.py
"""

import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
import streamlit as st

from data_prep import SUBJECT_TABLE_CSV
from orchestrator_agent import Orchestrator
from uncertainty_agent import UncertaintyAgent
from imaging_agent import image_to_tensor
import explainability
import llm_narrative

st.set_page_config(page_title="Thermal Triage Pipeline", page_icon="🌡️", layout="wide")

TIER_EMOJI = {"confident": "🟢", "uncertain": "🟡", "escalate": "🔴"}
TIER_LABEL = {"confident": "Auto-Classify", "uncertain": "Uncertainty Triage",
              "escalate": "Disagreement Resolution"}
TIER_LABEL_SHORT = {"confident": "Auto-Classify", "uncertain": "Triage", "escalate": "Escalate"}


# ─────────────────────────────────────────────
# Cached resources (loaded once per server process)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Imaging + Texture + Metadata agents…")
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_resource(show_spinner="Loading uncertainty stack (BNN + 5-model ensemble + OOD)…")
def get_uncertainty_agent() -> UncertaintyAgent:
    return UncertaintyAgent()


@st.cache_data
def get_subjects() -> pd.DataFrame:
    return pd.read_csv(SUBJECT_TABLE_CSV)


# ─────────────────────────────────────────────
# Cached per-subject computations
# (leading underscore on the agent params tells st.cache_data not to hash
# them — they're already singleton cache_resource objects)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_route(_orch: Orchestrator, subject_id: str) -> dict:
    return asdict(_orch.route(subject_id))


@st.cache_data(show_spinner=False)
def cached_uncertainty(_ua: UncertaintyAgent, subject_id: str) -> dict:
    return asdict(_ua.assess(subject_id))


@st.cache_data(show_spinner=False)
def cached_gradcam(_imaging, subject_id: str):
    return explainability.grad_cam_pp(_imaging, subject_id)   # (orig_u8, cam, p) | None


@st.cache_data(show_spinner=False)
def cached_shap(subject_id: str):
    return explainability.texture_shap_for_subject(subject_id, top_k=10)


@st.cache_data(show_spinner=False)
def cached_counterfactual(subject_id: str) -> dict:
    return asdict(explainability.texture_counterfactual(subject_id))


@st.cache_data(show_spinner=False)
def cached_attention(_imaging, subject_id: str):
    pred = _imaging.predict(subject_id)
    if not pred.available or not pred.view_attention:
        return None
    views = _imaging.by_subject[subject_id]
    if len(views) > _imaging.max_views:
        idx = np.linspace(0, len(views) - 1, _imaging.max_views).round().astype(int)
        views = [views[j] for j in idx]
    return [v.view for v in views], pred.view_attention, pred.probability


def gradcam_overlay_image(orig_u8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_resized = np.array(F.interpolate(
        torch.tensor(cam)[None, None], size=orig_u8.shape,
        mode="bilinear", align_corners=False)[0, 0])
    heat = cm.jet(cam_resized)[..., :3]
    base = np.stack([orig_u8] * 3, axis=-1).astype(np.float32) / 255.0
    return np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)


# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────

st.title("🌡️ Thermal Triage Pipeline")
st.caption("Imaging + Texture + Metadata agents → weighted-consensus routing → "
           "uncertainty quantification → explainability. Research prototype, not a diagnostic device.")

mode = st.sidebar.radio("Mode", ["Browse dataset", "Upload thermogram"])

# ───────────────────────── Browse dataset ─────────────────────────
if mode == "Browse dataset":
    subjects_df = get_subjects()

    st.sidebar.markdown("### Filter")
    datasets = sorted(subjects_df["dataset"].unique())
    dataset_sel = st.sidebar.multiselect("Dataset", datasets, default=datasets)
    label_sel = st.sidebar.multiselect(
        "Label", [0, 1], default=[0, 1],
        format_func=lambda x: "Healthy" if x == 0 else "Patient")

    filtered = subjects_df[subjects_df["dataset"].isin(dataset_sel) &
                            subjects_df["label"].isin(label_sel)]
    st.sidebar.caption(f"{len(filtered)} of {len(subjects_df)} subjects match")

    if filtered.empty:
        st.warning("No subjects match the current filter.")
        st.stop()

    subject_id = st.sidebar.selectbox("Subject", filtered["subject_id"].tolist())
    true_label = int(subjects_df.set_index("subject_id").loc[subject_id, "label"])

    orch = get_orchestrator()
    ua = get_uncertainty_agent()

    with st.spinner("Running agents…"):
        route = cached_route(orch, subject_id)
        unc = cached_uncertainty(ua, subject_id)

    # ── Header: routing decision ──────────────────────────────
    tier = route["tier"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routing tier", f"{TIER_EMOJI[tier]} {TIER_LABEL_SHORT[tier]}")
    c2.metric("C_final", f"{route['c_final']:.2f}")
    pred_txt = "—" if route["predicted_label"] is None else \
        ("Patient" if route["predicted_label"] == 1 else "Healthy")
    c3.metric("Predicted", pred_txt)
    c4.metric("True label", "Healthy" if true_label == 0 else "Patient")

    if route["predicted_label"] is not None:
        if route["predicted_label"] == true_label:
            st.success("Matches ground truth.")
        else:
            st.error("Does not match ground truth — see the expert breakdown below.")
    else:
        st.info(f"No auto-decision — routed to **{TIER_LABEL[tier]}** instead of a direct call.")

    st.divider()

    # ── Expert breakdown ───────────────────────────────────────
    st.subheader("Expert outputs")
    e1, e2, e3 = st.columns(3)
    e1.metric("Imaging — P(patient)",
              f"{route['p_imaging']:.2f}" if route["p_imaging"] is not None else "n/a")
    e2.metric("Texture — P(patient)",
              f"{route['p_texture']:.2f}" if route["p_texture"] is not None else "n/a")
    e3.metric("Metadata risk",
              f"{route['p_metadata_malignancy']:.2f}" if route["p_metadata_malignancy"] is not None
              else "n/a (non-Mendeley)")

    if route["risk_annotation"]:
        st.caption(route["risk_annotation"])
    if route["resolution_protocol"]:
        with st.expander("Disagreement resolution protocol", expanded=(tier == "escalate")):
            for step in route["resolution_protocol"]:
                st.write(f"- {step}")

    st.divider()

    # ── Uncertainty ─────────────────────────────────────────────
    st.subheader("Uncertainty quantification")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Tier", unc["tier"].replace("_", " "))
    u2.metric("Uncertainty", f"{unc['u_combined']:.3f}")
    u3.metric("OOD?", "Yes" if unc["is_ood"] else "No")
    u4.metric("Mean P(patient)", f"{unc['p_mean']:.2f}")

    with st.expander("Per-source uncertainty (variance across stochastic passes / ensemble members)"):
        s1, s2, s3 = st.columns(3)
        s1.metric("Bayesian NN posterior", f"{unc['u_bayesian']:.4f}" if unc["u_bayesian"] is not None else "n/a")
        s2.metric("MC-Dropout (T=30)", f"{unc['u_mc']:.4f}" if unc["u_mc"] is not None else "n/a")
        s3.metric("Deep Ensemble (M=5)", f"{unc['u_ensemble']:.4f}" if unc["u_ensemble"] is not None else "n/a")
        st.caption("MC-Dropout and the BNN redraw stochastic samples on every run — revisiting "
                   "this subject can shift these slightly. Deep Ensemble is deterministic.")

    st.divider()

    # ── Explainability ──────────────────────────────────────────
    st.subheader("Explainability")
    tab_gc, tab_shap, tab_attn, tab_cf, tab_note = st.tabs(
        ["Grad-CAM++", "Texture SHAP", "Multi-view attention", "Counterfactual", "Case narrative"])

    imaging = orch.imaging

    with tab_gc:
        result = cached_gradcam(imaging, subject_id)
        if result is None:
            st.info("Imaging data unavailable for this subject.")
        else:
            orig, cam, p = result
            overlay = gradcam_overlay_image(orig, cam)
            colA, colB = st.columns(2)
            colA.image(orig, caption="Thermogram (first view)", clamp=True, use_container_width=True)
            colB.image(overlay, caption=f"Grad-CAM++ — P(patient)={p:.2f}", use_container_width=True)
            st.caption("Heatmap from the ResNet34 backbone's last conv layer (Chattopadhay et al., "
                       "2018), backpropagated from the predicted class.")

    with tab_shap:
        shap_vals = cached_shap(subject_id)
        if not shap_vals:
            st.info("Texture data unavailable for this subject.")
        else:
            shap_df = pd.DataFrame(shap_vals, columns=["feature", "shap_value"]).set_index("feature")
            st.bar_chart(shap_df)
            st.caption("Positive = pushes the Texture Agent toward 'patient'; "
                       "negative = pushes toward 'healthy'.")

    with tab_attn:
        att = cached_attention(imaging, subject_id)
        if not att:
            st.info("No attention data for this subject.")
        else:
            names, weights, p = att
            att_df = pd.DataFrame(
                {"view": [f"{n} #{i}" for i, n in enumerate(names)], "attention": weights}
            ).set_index("view")
            st.bar_chart(att_df)
            st.caption("Learned per-view weight from the Imaging Agent's attention-pooling head — "
                       "this architecture's analog of the diagram's Attention Rollout (see "
                       "explainability.py's docstring for why: no ViT layers to roll out through).")

    with tab_cf:
        cf = cached_counterfactual(subject_id)
        if not cf["found"]:
            st.info(f"No single-feature change (searched the top 8 by SHAP) flips the Texture "
                     f"Agent's call within the observed dataset range (original P={cf['original_p']:.2f}). "
                     f"That itself is informative — the prediction isn't resting on one fragile feature.")
        else:
            st.write(f"Changing **`{cf['feature']}`** from `{cf['original_value']:.4f}` to "
                     f"`{cf['counterfactual_value']:.4f}` (holding every other feature fixed) "
                     f"would flip the Texture Agent's call — original P(patient)={cf['original_p']:.2f}.")

    with tab_note:
        case = dict(route)
        if shap_vals := cached_shap(subject_id):
            case["top_shap_feature"] = shap_vals[0][0]
        narrative = llm_narrative.generate_case_narrative(case)
        st.info(f"**[{narrative['source']}]**  {narrative['text']}")
        if narrative["source"] == "template":
            st.caption("No `GOOGLE_API_KEY` / `GROQ_API_KEY` in the environment — showing the "
                       "deterministic template. See llm_narrative.py's docstring for a free key.")

# ───────────────────────── Upload thermogram ─────────────────────────
else:
    st.subheader("Quick single-image check — Imaging Agent only")
    st.caption("Simplified path: skips the dataset-specific ROI harmonization in "
               "thermal_pipeline (2).py and the Texture/Metadata/Orchestrator agents, which need "
               "full per-subject multi-view context a single upload can't supply. For the faithful "
               "full pipeline, use **Browse dataset**.")

    uploaded = st.file_uploader("Upload a grayscale thermogram (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error("Could not decode that file as an image.")
        else:
            orch = get_orchestrator()
            imaging = orch.imaging
            tensor = image_to_tensor(img.astype(np.float32), imaging.img_size, augment=False)
            batch = tensor.unsqueeze(0)

            activations, gradients = {}, {}
            h1 = imaging.model.backbone.layer4.register_forward_hook(
                lambda m, i, o: activations.__setitem__("v", o))
            h2 = imaging.model.backbone.layer4.register_full_backward_hook(
                lambda m, gi, go: gradients.__setitem__("v", go[0]))
            try:
                imaging.model.zero_grad()
                logits, _ = imaging.model(batch, [1])
                p = float(torch.softmax(logits, dim=1)[0, 1].item())
                target_class = int(p >= imaging.threshold)
                logits[0, target_class].backward()
                cam = explainability._grad_cam_pp_map(activations["v"][0], gradients["v"][0])
            finally:
                h1.remove(); h2.remove()

            mn, mx = img.min(), img.max()
            orig_u8 = ((img.astype(np.float32) - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
            overlay = gradcam_overlay_image(orig_u8, cam)

            label = "Patient" if p >= imaging.threshold else "Healthy"
            st.metric("Imaging Agent prediction", label,
                      f"P(patient)={p:.2f}  (threshold={imaging.threshold:.2f})")
            colA, colB = st.columns(2)
            colA.image(orig_u8, caption="Uploaded thermogram", clamp=True, use_container_width=True)
            colB.image(overlay, caption="Grad-CAM++", use_container_width=True)
