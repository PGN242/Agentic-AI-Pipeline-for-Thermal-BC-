"""
Metadata Agent — clinical data processor.

Data reality check (read before trusting this module's numbers):

  Diagnostics.xlsx (Mendeley dataset only) has Age, Weight, Height, Temp and
  a Left/Right diagnosis code (N = normal, PB = benign pathology,
  PM = malignant / carcinoma) for each of the 119 subjects. Every single row
  has a finding on at least one side — there are zero fully-healthy (N/N)
  subjects in this table, and no clinical metadata exists at all for the
  DMR-IR or Kaggle subjects.

  That means clinical metadata CANNOT be used to train a healthy-vs-patient
  classifier (the label the rest of the pipeline uses) — there is no
  negative class to learn from. What the data DOES support is a real,
  non-trivial task: benign-vs-malignant risk stratification among patients
  who already have a finding. That is what `MetadataAgent` actually trains
  and predicts (`malignancy_risk`).

  The Orchestrator treats this as an auxiliary expert: it abstains
  (confidence 0) for every non-Mendeley subject and for Mendeley subjects
  with missing fields, and its opinion is used to annotate/escalate rather
  than to vote on the primary healthy/patient decision, which metadata
  alone structurally cannot answer.

  `risk_heuristic` is a simple, explicitly-labeled age/BMI-based proxy
  inspired by population breast-cancer risk trends (older age, higher BMI
  correlate with increased risk in epidemiological studies). It is NOT a
  clinically validated Gail model — the real Gail model needs family
  history, biopsy history, menarche/first-birth age, and race/ethnicity,
  none of which exist in this dataset. It is provided purely as an
  interpretable secondary signal alongside the trained classifier.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import joblib

from data_prep import DIAGNOSTICS_XLSX

ARTIFACT_PATH = r"C:/PhD/Implementation/artifacts/metadata_agent.joblib"

FEATURE_NAMES = ["age", "bmi", "weight_kg", "height_cm", "temp_c"]


# ─────────────────────────────────────────────
# Load + featurize Diagnostics.xlsx
# ─────────────────────────────────────────────

def _find_temp_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower().startswith("temp"):
            return c
    raise KeyError("No Temp(*C) column found in Diagnostics.xlsx")


def load_clinical_table(path: str = DIAGNOSTICS_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path)
    temp_col = _find_temp_col(df)

    out = pd.DataFrame()
    out["subject_id"] = "Mendeley::" + df["Image"].astype(str).str.strip()
    out["age"]        = df["Age(years)"].astype(float)
    out["weight_kg"]  = df["Weight (Kg)"].astype(float)
    out["height_cm"]  = df["Height(cm)"].astype(float)
    out["temp_c"]     = df[temp_col].astype(float)
    out["bmi"]        = out["weight_kg"] / ((out["height_cm"] / 100.0) ** 2)

    left  = df["Left"].astype(str).str.strip().str.upper()
    right = df["Right"].astype(str).str.strip().str.upper()
    out["malignant"] = ((left == "PM") | (right == "PM")).astype(int)
    out["side_finding"] = np.where(
        (left == "PM") | (right == "PM"), "malignant",
        np.where((left == "PB") | (right == "PB"), "benign", "unknown"))

    out = out.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    return out


def risk_heuristic(age: float, bmi: float) -> float:
    """
    Simple, documented age/BMI proxy in [0, 1] — see module docstring for
    why this is NOT the clinical Gail model. Centered so that ~50th
    percentile age/BMI in this dataset (~48y, ~29 BMI) sits near 0.5.
    """
    z = 0.045 * (age - 48.0) + 0.05 * (bmi - 29.0)
    return float(1.0 / (1.0 + np.exp(-z)))


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(save_path: str = ARTIFACT_PATH, seed: int = 42) -> Dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

    from agent_utils import calibrate_threshold, balanced_sample_weights

    df = load_clinical_table()
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["malignant"].to_numpy()

    n_pos = int(y.sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def _make_model():
        return GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.8, random_state=seed)

    # Manual OOF loop (rather than cross_val_predict) so we control
    # per-fold sample_weight directly, avoiding sklearn's metadata-routing
    # API churn across versions.
    oof_proba = np.zeros(len(y), dtype=np.float64)
    for tr_idx, te_idx in cv.split(X, y):
        fold_model = _make_model()
        fold_model.fit(X[tr_idx], y[tr_idx], sample_weight=balanced_sample_weights(y[tr_idx]))
        oof_proba[te_idx] = fold_model.predict_proba(X[te_idx])[:, 1]

    threshold = calibrate_threshold(y, oof_proba)
    oof_pred = (oof_proba >= threshold).astype(int)

    metrics = {
        "n_subjects": len(df),
        "n_malignant": n_pos,
        "n_benign": len(df) - n_pos,
        "cv_auc": float(roc_auc_score(y, oof_proba)) if 0 < n_pos < len(df) else float("nan"),
        "cv_accuracy": float(accuracy_score(y, oof_pred)),
        "cv_balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "cv_f1": float(f1_score(y, oof_pred)),
        "decision_threshold": threshold,
    }

    # Final model fit on all available clinical rows for deployment use.
    model = _make_model()
    model.fit(X, y, sample_weight=balanced_sample_weights(y))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "threshold": threshold}, save_path)

    print("[MetadataAgent] Trained benign-vs-malignant risk model "
          "(auxiliary — cannot see healthy subjects):")
    for k, v in metrics.items():
        print(f"    {k:20s}: {v}")
    print(f"[MetadataAgent] Saved -> {save_path}")
    return metrics


# ─────────────────────────────────────────────
# Inference API
# ─────────────────────────────────────────────

@dataclass
class MetadataPrediction:
    subject_id:      str
    available:       bool
    malignancy_risk: Optional[float] = None   # P(malignant | already has a finding)
    risk_heuristic:  Optional[float] = None   # age/BMI proxy, see docstring
    confidence:      float = 0.0              # distance from the calibrated threshold, 0 when abstaining
    features:        Optional[Dict[str, float]] = None


class MetadataAgent:
    """
    Loads the trained benign/malignant risk model and answers per-subject
    queries. Abstains (available=False, confidence=0) for any subject_id
    outside the Mendeley clinical table — which is most of the combined
    dataset (DMR-IR, Kaggle have no clinical metadata at all).
    """

    def __init__(self, artifact_path: str = ARTIFACT_PATH):
        if not os.path.exists(artifact_path):
            train(artifact_path)
        blob = joblib.load(artifact_path)
        self.model = blob["model"]
        self.feature_names = blob["feature_names"]
        self.threshold = blob.get("threshold", 0.5)
        self._table = load_clinical_table().set_index("subject_id")

    def predict(self, subject_id: str) -> MetadataPrediction:
        if subject_id not in self._table.index:
            return MetadataPrediction(subject_id=subject_id, available=False)

        row = self._table.loc[subject_id]
        feats = {k: float(row[k]) for k in self.feature_names}
        x = np.array([[feats[k] for k in self.feature_names]])
        p = float(self.model.predict_proba(x)[0, 1])
        heuristic = risk_heuristic(feats["age"], feats["bmi"])

        from agent_utils import signed_margin
        confidence = abs(signed_margin(p, self.threshold))

        return MetadataPrediction(
            subject_id=subject_id, available=True,
            malignancy_risk=p, risk_heuristic=heuristic,
            confidence=confidence, features=feats)

    def predict_batch(self, subject_ids: List[str]) -> pd.DataFrame:
        rows = [self.predict(sid).__dict__ for sid in subject_ids]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    train()
