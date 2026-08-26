"""
Orchestrator Agent — Dynamic Confidence Routing Mechanism.

Implements the diagram's mechanism precisely:

  Step 1 — Weighted Consensus Calculation
      C_final = sum(w_i * c_i), where w_i is agent i's HISTORICAL ACCURACY
      (its own VALIDATION-split balanced accuracy, read from its trained
      artifact — never the test split, so this weight can't leak into the
      numbers it's later evaluated against) and c_i is that agent's own
      per-sample confidence (its calibrated margin, in [0, 1]).

      When Imaging and Texture agree on the label, C_final sums both
      weighted-confidence terms. When they disagree, only the winning
      side's term counts, so disagreement mechanically produces a lower
      C_final — exactly the "agreement score" the diagram calls for,
      without a separately-fitted (and, on this dataset's ~15-subject
      minority class, easily overfit — see git history of this file)
      gating model.

  Step 2 — Disagreement Check, three tiers:
      C > 0.85            -> Auto-Classify, direct output (+ Grad-CAM++
                              reference for that subject, see explainability.py)
      0.60 <= C <= 0.85    -> Uncertainty Triage: pull the Bayesian
                              uncertainty score from uncertainty_agent.py,
                              flag for radiologist
      C < 0.60             -> Disagreement Resolution, structured protocol:
                              1. identify conflicting agents
                              2. weight by historical accuracy (done above)
                              3. request supplementary features (pull the
                                 Metadata Agent's malignancy risk)
                              4. MC-Dropout re-evaluation
                              5. escalate if still unresolved

Metadata Agent stays an annotation / supplementary-feature input, not a
primary voter — see metadata_agent.py's docstring: its training data has
zero healthy subjects, so it structurally cannot vote on the healthy/
patient question the other two agents answer.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import joblib

import imaging_agent
import texture_classifier
from data_prep import SUBJECT_TABLE_CSV, group_stratified_split
from imaging_agent import ImagingAgent
from texture_classifier import TextureExpert
from metadata_agent import MetadataAgent

AUTO_CLASSIFY_THRESH = 0.85
DISAGREEMENT_THRESH = 0.60


@dataclass
class OrchestratorResult:
    subject_id: str
    p_imaging:  Optional[float]
    p_texture:  Optional[float]
    p_metadata_malignancy: Optional[float]
    label_imaging: Optional[int]
    label_texture: Optional[int]
    c_final:    float                  # weighted-consensus agreement score, Step 1
    combined_label: Optional[int]      # the Step-1 weighted-consensus label, regardless of tier
    tier:       str                    # "confident" | "uncertain" | "escalate"
    predicted_label: Optional[int]     # 0=healthy, 1=patient (None unless tier == "confident")
    risk_annotation: Optional[str]
    resolution_protocol: Optional[List[str]] = field(default=None)
    bayesian_uncertainty: Optional[float] = None
    is_ood: Optional[bool] = None


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class Orchestrator:
    def __init__(self, use_uncertainty: bool = True):
        self.imaging = ImagingAgent()
        self.texture = TextureExpert()
        self.metadata = MetadataAgent()
        self.w_imaging, self.w_texture = self._historical_weights()
        self._use_uncertainty = use_uncertainty
        self._uncertainty = None
        print(f"[Orchestrator] historical-accuracy weights (val split): "
              f"imaging={self.w_imaging:.3f} texture={self.w_texture:.3f}")

    @staticmethod
    def _historical_weights() -> (float, float):
        img_blob = torch.load(imaging_agent.ARTIFACT_PATH, map_location="cpu", weights_only=False)
        w_img = img_blob["metrics"].get("best_val_balanced_accuracy", 0.5)
        tex_blob = joblib.load(texture_classifier.ARTIFACT_PATH)
        w_tex = tex_blob.get("metrics", {}).get("val_balanced_accuracy", 0.5)
        total = w_img + w_tex
        return w_img / total, w_tex / total

    def _uncertainty_agent(self):
        if self._uncertainty is None:
            from uncertainty_agent import UncertaintyAgent
            self._uncertainty = UncertaintyAgent()
        return self._uncertainty

    # ── Step 1: weighted consensus ────────────────────────────
    def _consensus(self, ip, tp):
        """Returns (c_final, combined_label, agreeing_agents, conflicting_agents)."""
        if ip.available and tp.available:
            label_img = int(ip.probability >= self.imaging.threshold)
            label_tex = int(tp.probability >= self.texture.threshold)
            term_img = self.w_imaging * ip.confidence
            term_tex = self.w_texture * tp.confidence
            if label_img == label_tex:
                return term_img + term_tex, label_img, ["imaging", "texture"], []
            if term_img >= term_tex:
                return term_img, label_img, ["imaging"], ["texture"]
            return term_tex, label_tex, ["texture"], ["imaging"]
        if ip.available:
            label_img = int(ip.probability >= self.imaging.threshold)
            return ip.confidence, label_img, ["imaging"], []
        if tp.available:
            label_tex = int(tp.probability >= self.texture.threshold)
            return tp.confidence, label_tex, ["texture"], []
        return 0.0, None, [], []

    def route(self, subject_id: str) -> OrchestratorResult:
        ip = self.imaging.predict(subject_id)
        tp = self.texture.predict(subject_id)
        mp = self.metadata.predict(subject_id)

        c_final, combined_label, agreeing, conflicting = self._consensus(ip, tp)

        risk_note = None
        if mp.available:
            risk_note = (f"metadata malignancy_risk={mp.malignancy_risk:.2f} "
                         f"(risk_heuristic={mp.risk_heuristic:.2f})")

        label_img = int(ip.probability >= self.imaging.threshold) if ip.available else None
        label_tex = int(tp.probability >= self.texture.threshold) if tp.available else None

        resolution: Optional[List[str]] = None
        bayes_u, is_ood = None, None

        if combined_label is None:
            tier, pred = "escalate", None
            resolution = ["No expert produced a usable prediction for this subject."]
        elif c_final > AUTO_CLASSIFY_THRESH:
            tier, pred = "confident", combined_label
        elif c_final >= DISAGREEMENT_THRESH:
            tier, pred = "uncertain", None
            if self._use_uncertainty:
                try:
                    u = self._uncertainty_agent().assess(subject_id)
                    bayes_u, is_ood = u.u_combined, u.is_ood
                except Exception as e:
                    bayes_u = None
        else:
            tier, pred = "escalate", None
            resolution = [
                f"1. conflicting agents: {conflicting or ['none — both experts agree but jointly under-confident']}",
                f"2. historical-accuracy weights: imaging={self.w_imaging:.2f}, texture={self.w_texture:.2f}",
                (f"3. supplementary feature — metadata malignancy_risk={mp.malignancy_risk:.2f}"
                 if mp.available else "3. supplementary feature requested — metadata unavailable for this subject"),
            ]
            if self._use_uncertainty:
                try:
                    mc = self.imaging.mc_dropout_probs(subject_id, T=30)
                    if mc is not None:
                        resolution.append(f"4. MC-Dropout re-evaluation: mean={mc.mean():.2f} std={mc.std():.2f}")
                        u = self._uncertainty_agent().assess(subject_id)
                        bayes_u, is_ood = u.u_combined, u.is_ood
                except Exception:
                    resolution.append("4. MC-Dropout re-evaluation: unavailable")
            resolution.append("5. still unresolved -> escalated for radiologist re-analysis")

        return OrchestratorResult(
            subject_id=subject_id,
            p_imaging=ip.probability if ip.available else None,
            p_texture=tp.probability if tp.available else None,
            p_metadata_malignancy=mp.malignancy_risk if mp.available else None,
            label_imaging=label_img, label_texture=label_tex,
            c_final=c_final, combined_label=combined_label, tier=tier, predicted_label=pred,
            risk_annotation=risk_note, resolution_protocol=resolution,
            bayesian_uncertainty=bayes_u, is_ood=is_ood)

    def route_batch(self, subject_ids: List[str]) -> pd.DataFrame:
        return pd.DataFrame([asdict(self.route(sid)) for sid in subject_ids])


# ─────────────────────────────────────────────
# Evaluation on the held-out test split
# ─────────────────────────────────────────────

def evaluate(seed: int = 42, use_uncertainty: bool = True) -> Dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    label_map = subjects_df.set_index("subject_id")["label"].to_dict()
    _, _, test_ids = group_stratified_split(subjects_df, seed=seed)

    orch = Orchestrator(use_uncertainty=use_uncertainty)
    results = orch.route_batch(test_ids)
    results["true_label"] = results["subject_id"].map(label_map)

    report: Dict = {
        "n_test_subjects": len(results),
        "w_imaging": orch.w_imaging, "w_texture": orch.w_texture,
    }
    for tier, diagram_target in (("confident", 60.0), ("uncertain", 20.0), ("escalate", 15.0)):
        sub = results[results["tier"] == tier]
        report[f"tier_{tier}_n"] = len(sub)
        report[f"tier_{tier}_pct"] = round(100 * len(sub) / max(len(results), 1), 1)
        report[f"tier_{tier}_diagram_target_pct"] = diagram_target

    confident = results[results["tier"] == "confident"]
    if len(confident):
        report["confident_accuracy"] = float(accuracy_score(confident["true_label"], confident["predicted_label"]))
        report["confident_balanced_accuracy"] = float(
            balanced_accuracy_score(confident["true_label"], confident["predicted_label"]))

    # Baseline: trust the Step-1 weighted-consensus label with no routing at all.
    report["always_decide_accuracy"] = float(accuracy_score(results["true_label"], results["combined_label"]))
    report["always_decide_balanced_accuracy"] = float(
        balanced_accuracy_score(results["true_label"], results["combined_label"]))

    print("[Orchestrator] Test-split routing report:")
    for k, v in report.items():
        print(f"    {k:32s}: {v}")
    results.to_csv(r"C:/PhD/Implementation/artifacts/orchestrator_test_results.csv", index=False)
    return report


if __name__ == "__main__":
    evaluate()
