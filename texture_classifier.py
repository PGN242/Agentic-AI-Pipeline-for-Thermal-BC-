"""
Texture Agent classifier head.

Phase 2 (texture_agent.py) extracts a large, fixed, named descriptor per
IMAGE (GLCM/LBP/Gabor/gradient/statistical families). This module:

  1. Runs that extractor over every image in processed_all/ (once — the
     per-image feature table is cached to CSV since extraction is the
     expensive step).
  2. Mean-pools per-image features up to one row per SUBJECT (a subject may
     have 1-8 views/frames after data_prep's subsampling).
  3. Trains a shallow classifier (GradientBoosting, with univariate feature
     selection to keep the model sane relative to the ~230-subject sample
     size) for the binary healthy/patient task, evaluated on the same
     group-stratified subject split used everywhere else in the pipeline.
  4. Exposes TextureExpert.predict(subject_id) -> probability + confidence
     for the Orchestrator.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from data_prep import (PROCESSED_ROOT, subject_id_from_patient_id,
                        SUBJECT_TABLE_CSV)
from texture_agent import TextureAgent, TextureConfig, load_processed_images

FEATURES_CSV = r"C:/PhD/Implementation/artifacts/texture_features_per_image.csv"
SUBJECT_FEATURES_CSV = r"C:/PhD/Implementation/artifacts/texture_features_per_subject.csv"
ARTIFACT_PATH = r"C:/PhD/Implementation/artifacts/texture_agent.joblib"


# ─────────────────────────────────────────────
# Step 1 — per-image feature extraction (cached)
# ─────────────────────────────────────────────

def extract_per_image_features(processed_dir: str = PROCESSED_ROOT,
                                 out_csv: str = FEATURES_CSV,
                                 force: bool = False) -> pd.DataFrame:
    if os.path.exists(out_csv) and not force:
        print(f"[TextureExpert] Using cached per-image features: {out_csv}")
        return pd.read_csv(out_csv)

    agent = TextureAgent(TextureConfig())
    samples = load_processed_images(processed_dir)
    feats = agent.extract_batch(samples)
    df = agent.to_dataframe(feats)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[TextureExpert] Extracted {len(df)} rows -> {out_csv}")
    return df


# ─────────────────────────────────────────────
# Step 2 — mean-pool to subject level
# ─────────────────────────────────────────────

def pool_to_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["valid"]].copy()
    df["subject_id"] = [subject_id_from_patient_id(pid, ds)
                         for pid, ds in zip(df["patient_id"], df["dataset"])]

    meta_cols = {"patient_id", "dataset", "view", "label", "source_path", "valid", "subject_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    pooled = df.groupby("subject_id")[feat_cols].mean()
    label = df.groupby("subject_id")["label"].max()   # patient(1) wins if inconsistent
    out = pooled.join(label).reset_index()
    return out


# ─────────────────────────────────────────────
# Step 3 — train
# ─────────────────────────────────────────────

def train(force_extract: bool = False, seed: int = 42) -> Dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

    from data_prep import group_stratified_split
    from agent_utils import calibrate_threshold, balanced_sample_weights

    per_image = extract_per_image_features(force=force_extract)
    subj = pool_to_subject_level(per_image)
    subj = subj.dropna(axis=1, how="any")  # drop any feature that's NaN for some subject
    subj.to_csv(SUBJECT_FEATURES_CSV, index=False)

    feat_cols = [c for c in subj.columns if c not in ("subject_id", "label")]
    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    train_ids, val_ids, test_ids = group_stratified_split(subjects_df, seed=seed)

    def subset(ids):
        d = subj[subj["subject_id"].isin(ids)]
        return d[feat_cols].to_numpy(dtype=np.float64), d["label"].to_numpy(), d["subject_id"].to_numpy()

    X_tr, y_tr, _ = subset(train_ids)
    X_va, y_va, _ = subset(val_ids)
    X_te, y_te, _ = subset(test_ids)

    k = min(40, X_tr.shape[1])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=seed)),
    ])
    # GradientBoostingClassifier has no class_weight param -> balance via
    # per-sample weights (inverse class frequency). Without this the model
    # chases the majority "patient" class and a fixed 0.5 threshold collapses
    # onto near-chance balanced accuracy for the 19-subject healthy class.
    pipe.fit(X_tr, y_tr, clf__sample_weight=balanced_sample_weights(y_tr))

    # Calibrate the decision threshold on VAL (held out from the fit above)
    # rather than assuming 0.5 — the fix for the imbalance-driven threshold
    # collapse noted above.
    threshold = (calibrate_threshold(y_va, pipe.predict_proba(X_va)[:, 1])
                 if len(y_va) else 0.5)

    def eval_split(X, y, name):
        if len(y) == 0:
            return {}
        p = pipe.predict_proba(X)[:, 1]
        pred = (p >= threshold).astype(int)
        m = {
            f"{name}_n": len(y),
            f"{name}_accuracy": float(accuracy_score(y, pred)),
            f"{name}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            f"{name}_f1": float(f1_score(y, pred, zero_division=0)),
        }
        if len(set(y.tolist())) > 1:
            m[f"{name}_auc"] = float(roc_auc_score(y, p))
        return m

    metrics = {"decision_threshold": threshold}
    metrics.update(eval_split(X_tr, y_tr, "train"))
    metrics.update(eval_split(X_va, y_va, "val"))
    metrics.update(eval_split(X_te, y_te, "test"))

    # Refit on train+val for the deployed model; test stays untouched for reporting.
    # The threshold stays the one calibrated above (test must never leak in).
    X_trva = np.concatenate([X_tr, X_va]) if len(X_va) else X_tr
    y_trva = np.concatenate([y_tr, y_va]) if len(y_va) else y_tr
    pipe.fit(X_trva, y_trva, clf__sample_weight=balanced_sample_weights(y_trva))

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": feat_cols, "threshold": threshold,
                 "metrics": metrics}, ARTIFACT_PATH)

    print("[TextureExpert] Trained. Metrics:")
    for k_, v in metrics.items():
        print(f"    {k_:24s}: {v}")
    print(f"[TextureExpert] Saved -> {ARTIFACT_PATH}")
    return metrics


# ─────────────────────────────────────────────
# Inference API
# ─────────────────────────────────────────────

@dataclass
class TextureExpertPrediction:
    subject_id:  str
    available:   bool
    probability: Optional[float] = None   # P(patient)
    confidence:  float = 0.0


class TextureExpert:
    def __init__(self, artifact_path: str = ARTIFACT_PATH):
        if not os.path.exists(artifact_path):
            train()
        blob = joblib.load(artifact_path)
        self.pipe = blob["pipeline"]
        self.feature_names = blob["feature_names"]
        self.threshold = blob.get("threshold", 0.5)
        self._table = pd.read_csv(SUBJECT_FEATURES_CSV).set_index("subject_id")

    def predict(self, subject_id: str) -> TextureExpertPrediction:
        if subject_id not in self._table.index:
            return TextureExpertPrediction(subject_id=subject_id, available=False)
        row = self._table.loc[subject_id]
        x = row[self.feature_names].to_numpy(dtype=np.float64).reshape(1, -1)
        p = float(self.pipe.predict_proba(x)[0, 1])
        from agent_utils import signed_margin
        return TextureExpertPrediction(
            subject_id=subject_id, available=True,
            probability=p, confidence=abs(signed_margin(p, self.threshold)))

    def predict_batch(self, subject_ids: List[str]) -> pd.DataFrame:
        rows = [self.predict(sid).__dict__ for sid in subject_ids]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    train()
