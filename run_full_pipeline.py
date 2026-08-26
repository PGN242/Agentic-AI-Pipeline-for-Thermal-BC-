"""
End-to-end entry point for the full architecture in the diagram:

    Input: Multi-View Thermograms
      -> Preprocessing (thermal_pipeline (2).py)
      -> Feature Distribution Layer (data_prep.py -> processed_all/)
      -> Imaging Agent | Texture Agent | Metadata Agent   (parallel experts)
      -> Orchestrator Agent (dynamic gating, disagreement detection,
                              confidence routing)
      -> Uncertainty Quantification (BNN + MC-Dropout + Deep Ensemble + OOD)
      -> Explainability (Grad-CAM++, SHAP, attention, counterfactual)
      -> Confident Classification | Uncertainty Triage | Escalation

Each stage is idempotent: it loads its cached artifact if present, and only
retrains when the artifact is missing. Run with --force to retrain
everything from scratch, or --skip-ensemble to skip the ~35-40 min deep
ensemble training (uncertainty quantification then runs with 2 of its 3
sources instead of 3).
"""

import argparse
import json
import os

import pandas as pd

import data_prep
import metadata_agent
import texture_classifier
import imaging_agent
import orchestrator_agent
import uncertainty_agent
import explainability
import llm_narrative


def main(force: bool = False, skip_ensemble: bool = False):
    report = {}

    print("\n" + "=" * 70)
    print("STAGE 1/8 — Preprocessing + Feature Distribution Layer")
    print("=" * 70)
    if force or not os.path.exists(data_prep.SUBJECT_TABLE_CSV):
        subjects = data_prep.run()
    else:
        subjects = pd.read_csv(data_prep.SUBJECT_TABLE_CSV)
        print(f"[run_full_pipeline] Using cached subject table "
              f"({len(subjects)} subjects) -> {data_prep.SUBJECT_TABLE_CSV}")
    report["n_subjects"] = int(len(subjects))
    report["label_counts"] = subjects["label"].value_counts().to_dict()
    report["dataset_x_label"] = (
        subjects.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_dict())

    print("\n" + "=" * 70)
    print("STAGE 2/8 — Metadata Agent (clinical / tabular)")
    print("=" * 70)
    report["metadata_metrics"] = metadata_agent.train()

    print("\n" + "=" * 70)
    print("STAGE 3/8 — Texture Agent (GLCM/LBP/Gabor/gradient/statistical)")
    print("=" * 70)
    report["texture_metrics"] = texture_classifier.train(force_extract=force)

    print("\n" + "=" * 70)
    print("STAGE 4/8 — Imaging Agent (ResNet34 + multi-view attention fusion)")
    print("=" * 70)
    if force or not os.path.exists(imaging_agent.ARTIFACT_PATH):
        report["imaging_metrics"] = imaging_agent.train()
    else:
        import torch
        blob = torch.load(imaging_agent.ARTIFACT_PATH, map_location="cpu", weights_only=False)
        report["imaging_metrics"] = blob.get("metrics", {})
        print(f"[run_full_pipeline] Using cached Imaging Agent -> {imaging_agent.ARTIFACT_PATH}")

    print("\n" + "=" * 70)
    print("STAGE 5/8 — Deep Ensemble (M=5 Imaging Agents, Phase 3 uncertainty source)")
    print("=" * 70)
    import glob
    existing = glob.glob(f"{imaging_agent.ENSEMBLE_DIR}/member_*.pt")
    if skip_ensemble:
        print("[run_full_pipeline] --skip-ensemble: leaving deep ensemble as-is "
              f"({len(existing)} member(s) present).")
    elif force or len(existing) < len(imaging_agent.ENSEMBLE_SEEDS):
        report["ensemble_metrics"] = imaging_agent.train_ensemble()
    else:
        print(f"[run_full_pipeline] Using cached deep ensemble "
              f"({len(existing)} members) -> {imaging_agent.ENSEMBLE_DIR}")

    print("\n" + "=" * 70)
    print("STAGE 6/8 — Uncertainty Quantification (BNN + MC-Dropout + Ensemble + OOD)")
    print("=" * 70)
    if force or not os.path.exists(uncertainty_agent.OOD_ARTIFACT):
        uncertainty_agent.fit_ood_stats()
    if force or not os.path.exists(uncertainty_agent.BNN_ARTIFACT):
        report["bnn_metrics"] = uncertainty_agent.train_bnn()
    else:
        import torch as _torch
        report["bnn_metrics"] = _torch.load(
            uncertainty_agent.BNN_ARTIFACT, map_location="cpu", weights_only=False)["metrics"]
    report["uncertainty_report"] = uncertainty_agent.evaluate()

    print("\n" + "=" * 70)
    print("STAGE 7/8 — Orchestrator Agent (confidence routing, disagreement resolution)")
    print("=" * 70)
    report["orchestrator_report"] = orchestrator_agent.evaluate()

    print("\n" + "=" * 70)
    print("STAGE 8/8 — Explainability (Grad-CAM++, SHAP, attention, counterfactual) "
          "+ optional LLM narrative")
    print("=" * 70)
    from data_prep import group_stratified_split
    _, _, test_ids = group_stratified_split(subjects, seed=42)
    label_map = subjects.set_index("subject_id")["label"].to_dict()
    sample_ids = test_ids[:8]
    report["explainability"] = explainability.run_all(sample_ids, label_map)

    # Route the sample subjects fresh (in-memory objects) rather than reading
    # back the saved CSV — CSV round-tripping stringifies the
    # resolution_protocol list, which the narrative generator needs intact.
    from dataclasses import asdict
    orch = orchestrator_agent.Orchestrator()
    sample_rows = [asdict(orch.route(sid)) for sid in sample_ids]
    narratives = llm_narrative.generate_batch_narratives(sample_rows)
    report["narratives"] = narratives
    report["narrative_source"] = narratives[0]["source"] if narratives else "none"

    out_path = r"C:/PhD/Implementation/artifacts/final_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[run_full_pipeline] Full report saved -> {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain every stage from scratch")
    parser.add_argument("--skip-ensemble", action="store_true",
                         help="Skip deep-ensemble training (~35-40 min); use whatever's cached")
    args = parser.parse_args()
    main(force=args.force, skip_ensemble=args.skip_ensemble)
