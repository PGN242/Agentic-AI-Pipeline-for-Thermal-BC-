"""
Feature Distribution Layer — data preparation shared by all three expert agents.

Responsibilities:
  1. Load the three raw datasets (DMR-IR, Kaggle, Mendeley) through the
     Phase-1 loaders, tagging every sample with a *subject_id* that is
     stable across frames/sides/views (unlike ThermalSample.patient_id,
     which for DMR-IR/Kaggle bakes in the video-frame sequence number).
  2. Subsample the DMR-IR / Kaggle thermal "video" sequences — they are
     20+ near-duplicate frames per (subject, side) — down to a handful of
     evenly-spaced frames so downstream agents aren't trained on massively
     redundant data dominated by two of the three datasets.
  3. Run the Phase-1 preprocessing pipeline (normalize -> harmonize ->
     align -> ROI-extract) once and persist the result to a single
     `processed_all/<dataset>/<healthy|patient>/<file>.png` tree. Every
     expert agent (Imaging, Texture) reads from this same tree, which is
     the concrete form of the diagram's "Feature Distribution Layer".
  4. Build a subject-level table (subject_id, dataset, label, views) and a
     group-aware, class-stratified train/val/test split over *subjects*
     (never over frames — that would leak the same patient across splits).

The dataset composition is heavily imbalanced: only DMR-IR contributes
healthy controls. Kaggle and Mendeley are both patient-only datasets
(Benign/Malignant, no Normal class). Combined:

    healthy subjects : 28   (DMR-IR only)
    patient subjects : 231  (28 DMR-IR + 56 Kaggle + 119 Mendeley + 28 DMR-IR-cancerous... see report)

This module does not hide that imbalance — `build_subject_table` reports it,
and `group_stratified_split` uses stratified sampling + class weights
downstream rather than pretending the classes are balanced.
"""

import os
import re
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

# thermal_pipeline lives in a file with spaces/parens in its name.
_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("tp", str(_here / "thermal_pipeline (2).py"))
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

ThermalSample = tp.ThermalSample
ViewType = tp.ViewType
PipelineConfig = tp.PipelineConfig
ThermalPreprocessingPipeline = tp.ThermalPreprocessingPipeline
save_processed = tp.save_processed
load_pac_dataset = tp.load_pac_dataset
load_mendeley_dataset = tp.load_mendeley_dataset


# ─────────────────────────────────────────────
# Raw dataset roots (edit to match your machine)
# ─────────────────────────────────────────────

DATASET_ROOTS = {
    "DMR-IR":   r"C:\PhD\Datasets\BC DMR IR thermo",
    "Kaggle":   r"C:\PhD\Datasets\BC Kagggle thermo\breast-cancer-dataset",
    "Mendeley": r"C:\PhD\Datasets\BC Mendely thermo\Breast Thermography",
}
DIAGNOSTICS_XLSX = r"C:\PhD\Datasets\BC Mendely thermo\Breast Thermography\Diagnostics.xlsx"

PROCESSED_ROOT = r"C:/datasets/processed_all"
SUBJECT_TABLE_CSV = r"C:/PhD/Implementation/artifacts/subjects.csv"


# ─────────────────────────────────────────────
# subject_id — stable across frames / sides / views
# ─────────────────────────────────────────────

_DN_SUFFIX_RE = re.compile(r"_DN\d+$", re.IGNORECASE)


def subject_id_from_patient_id(patient_id: str, dataset: str) -> str:
    """
    Strip the frame-sequence suffix ("_DN0", "_DN17", ...) that the PAC
    loader bakes into ThermalSample.patient_id, so repeated video frames of
    the same subject collapse to one subject_id. Mendeley ids are already
    subject-level (IIR0019) and pass through unchanged.
    """
    base = _DN_SUFFIX_RE.sub("", patient_id)
    return f"{dataset}::{base}"


def derive_subject_id(sample: ThermalSample) -> str:
    return subject_id_from_patient_id(sample.patient_id, sample.dataset)


# ─────────────────────────────────────────────
# Frame subsampling (DMR-IR / Kaggle video sequences)
# ─────────────────────────────────────────────

def subsample_frames(samples: List[ThermalSample],
                      max_per_subject_view: int = 4) -> List[ThermalSample]:
    """
    Keep at most `max_per_subject_view` evenly-spaced frames per
    (subject_id, view). Mendeley subjects only ever have 1 frame per view
    so this is a no-op for them; DMR-IR/Kaggle collapse from ~20 to a few.
    """
    groups: Dict[Tuple[str, str], List[ThermalSample]] = defaultdict(list)
    for s in samples:
        key = (derive_subject_id(s), s.view.value)
        groups[key].append(s)

    kept: List[ThermalSample] = []
    for key, group in groups.items():
        n = len(group)
        if n <= max_per_subject_view:
            kept.extend(group)
            continue
        idx = np.linspace(0, n - 1, max_per_subject_view).round().astype(int)
        idx = sorted(set(idx.tolist()))
        kept.extend(group[i] for i in idx)
    return kept


# ─────────────────────────────────────────────
# Load + subsample all three raw datasets
# ─────────────────────────────────────────────

def load_all_raw(max_per_subject_view: int = 4) -> List[ThermalSample]:
    raw: List[ThermalSample] = []
    raw += load_pac_dataset(DATASET_ROOTS["DMR-IR"], dataset_name="DMR-IR")
    raw += load_pac_dataset(DATASET_ROOTS["Kaggle"], dataset_name="Kaggle")
    raw += load_mendeley_dataset(DATASET_ROOTS["Mendeley"])

    before = len(raw)
    raw = subsample_frames(raw, max_per_subject_view=max_per_subject_view)
    print(f"[data_prep] Subsampled {before} -> {len(raw)} frames "
          f"(max {max_per_subject_view}/subject/view)")

    n_unlabeled = sum(1 for s in raw if s.label is None)
    if n_unlabeled:
        print(f"[data_prep] Dropping {n_unlabeled} frames with unresolved label")
        raw = [s for s in raw if s.label is not None]
    return raw


# ─────────────────────────────────────────────
# Phase-1 preprocessing -> processed_all/
# ─────────────────────────────────────────────

def run_preprocessing(raw: List[ThermalSample],
                       output_dir: str = PROCESSED_ROOT) -> List[ThermalSample]:
    config = PipelineConfig(target_size=(320, 240), augment=False)
    pipeline = ThermalPreprocessingPipeline(config)

    processed, saved, blanks, errors = [], 0, 0, 0
    for i, sample in enumerate(raw):
        try:
            out = pipeline(sample, augment=False)
            path = save_processed(out, output_dir)
            if path is None:
                blanks += 1
            else:
                saved += 1
                processed.append(out)
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(raw)}] ERROR | {sample.source_path} | {e}")
        if (i + 1) % 200 == 0:
            print(f"  ...preprocessed {i+1}/{len(raw)}")

    print(f"[data_prep] Preprocessing done: saved={saved} blanks={blanks} errors={errors}")
    return processed


# ─────────────────────────────────────────────
# Subject table
# ─────────────────────────────────────────────

@dataclass
class SubjectRow:
    subject_id: str
    dataset:    str
    label:      int
    n_views:    int


def build_subject_table(processed: List[ThermalSample]) -> pd.DataFrame:
    per_subject: Dict[str, Dict] = {}
    for s in processed:
        sid = derive_subject_id(s)
        row = per_subject.setdefault(sid, {"dataset": s.dataset, "labels": set(), "n_views": 0})
        row["labels"].add(s.label)
        row["n_views"] += 1

    records = []
    for sid, row in per_subject.items():
        labels = row["labels"] - {None}
        label = max(labels) if labels else None   # patient(1) wins any inconsistency
        records.append(SubjectRow(sid, row["dataset"], label, row["n_views"]))

    df = pd.DataFrame([r.__dict__ for r in records]).sort_values("subject_id").reset_index(drop=True)
    return df


def save_subject_table(df: pd.DataFrame, path: str = SUBJECT_TABLE_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_prep] Subject table saved: {path}")
    print(df.groupby(["dataset", "label"]).size().unstack(fill_value=0))


# ─────────────────────────────────────────────
# Group-aware, class-stratified subject split
# ─────────────────────────────────────────────

def group_stratified_split(df: pd.DataFrame, val_frac: float = 0.15,
                            test_frac: float = 0.15, seed: int = 42
                            ) -> Tuple[List[str], List[str], List[str]]:
    """
    Split at the SUBJECT level (never at the frame level, to avoid leaking
    a patient's other frames/sides across train/val/test), stratified by
    label so the 28 healthy subjects are spread across all three splits.
    """
    from sklearn.model_selection import train_test_split

    ids = df["subject_id"].to_numpy()
    labels = df["label"].to_numpy()

    train_ids, rest_ids, train_y, rest_y = train_test_split(
        ids, labels, test_size=val_frac + test_frac, stratify=labels,
        random_state=seed)
    rel_test = test_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=rel_test, stratify=rest_y, random_state=seed)

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run(max_per_subject_view: int = 4) -> pd.DataFrame:
    raw = load_all_raw(max_per_subject_view=max_per_subject_view)
    processed = run_preprocessing(raw)
    subjects = build_subject_table(processed)
    save_subject_table(subjects)
    return subjects


if __name__ == "__main__":
    run()
