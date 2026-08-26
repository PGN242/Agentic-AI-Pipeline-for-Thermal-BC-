"""
Imaging Agent — CNN backbone + multi-view cross-attention fusion.

Architecture (matches the "Imaging Agent" box in the pipeline diagram):
  ResNet34 (ImageNet-pretrained, grayscale->3ch replication) extracts a
  512-d embedding per view. A subject may contribute 1-8 views/frames
  (anterior/oblique/lateral angles, or repeated DMR-IR/Kaggle frames of the
  same side) after data_prep's subsampling. An attention-pooling module
  ("multi-view cross-attention fusion") learns a per-view weight and
  combines them into one subject-level embedding, which a small MLP head
  classifies into healthy(0)/patient(1).

Runs CPU-only in this environment — image size and backbone are kept
modest (default 160x160, ResNet34) to keep a real training run tractable
without a GPU.
"""

import os
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_prep import (PROCESSED_ROOT, SUBJECT_TABLE_CSV,
                        subject_id_from_patient_id, group_stratified_split)
from texture_agent import load_processed_images, _LoadedSample

ARTIFACT_PATH = r"C:/PhD/Implementation/artifacts/imaging_agent.pt"
IMG_SIZE = 160
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────
# Image -> tensor pipeline
# ─────────────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """Thermal-aware augmentation, applied on the raw uint8 grayscale array."""
    h, w = img.shape
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < 0.4:
        delta = random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) + delta, 0, 255)
    if random.random() < 0.3:
        noise = np.random.normal(0, 6.0, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return img


def image_to_tensor(img: np.ndarray, img_size: int = IMG_SIZE,
                     augment: bool = False) -> torch.Tensor:
    img = img.astype(np.float32)
    if augment:
        img = _augment(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    rgb = np.stack([img, img, img], axis=-1)          # grayscale -> 3ch
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


# ─────────────────────────────────────────────
# Dataset — one item per SUBJECT, variable # of views
# ─────────────────────────────────────────────

class MultiViewSubjectDataset(Dataset):
    def __init__(self, subject_ids: List[str],
                 by_subject: Dict[str, List[_LoadedSample]],
                 img_size: int = IMG_SIZE, augment: bool = False,
                 max_views: int = 8):
        self.subject_ids = subject_ids
        self.by_subject = by_subject
        self.img_size = img_size
        self.augment = augment
        self.max_views = max_views

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, i):
        sid = self.subject_ids[i]
        views = self.by_subject[sid]
        if len(views) > self.max_views:
            idx = np.linspace(0, len(views) - 1, self.max_views).round().astype(int)
            views = [views[j] for j in idx]
        tensors = [image_to_tensor(v.image, self.img_size, self.augment) for v in views]
        label = views[0].label
        return tensors, label, sid


def collate_subjects(batch):
    imgs_flat, view_counts, labels, sids = [], [], [], []
    for tensors, label, sid in batch:
        imgs_flat.extend(tensors)
        view_counts.append(len(tensors))
        labels.append(label)
        sids.append(sid)
    images = torch.stack(imgs_flat, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, view_counts, labels, sids


def build_subject_index(processed_dir: str = PROCESSED_ROOT
                         ) -> Dict[str, List[_LoadedSample]]:
    samples = load_processed_images(processed_dir)
    by_subject: Dict[str, List[_LoadedSample]] = defaultdict(list)
    for s in samples:
        if s.label is None:
            continue
        sid = subject_id_from_patient_id(s.patient_id, s.dataset)
        by_subject[sid].append(s)
    return by_subject


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Learned per-view attention weights -> weighted-sum subject embedding."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, embeddings: torch.Tensor, view_counts: List[int]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outs, attn_all = [], []
        idx = 0
        for n in view_counts:
            chunk = embeddings[idx: idx + n]                 # [n, dim]
            scores = self.score(chunk).squeeze(-1)            # [n]
            weights = torch.softmax(scores, dim=0)             # [n]
            pooled = (weights.unsqueeze(-1) * chunk).sum(dim=0)
            outs.append(pooled)
            attn_all.append(weights.detach())
            idx += n
        return torch.stack(outs, dim=0), attn_all


class ImagingNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", n_classes: int = 2,
                 pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = 512
        elif backbone == "resnet34":
            net = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        net.fc = nn.Identity()
        self.backbone = net
        self.embed_dim = dim
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def embed(self, images: torch.Tensor, view_counts: List[int]):
        """Subject-level embedding (post attention-pool, pre classifier head)
        — the representation Mahalanobis OOD detection is computed on."""
        per_view = self.backbone(images)                       # [total_views, dim]
        subject_emb, attn = self.pool(per_view, view_counts)   # [B, dim]
        return subject_emb, attn

    def forward(self, images: torch.Tensor, view_counts: List[int]):
        subject_emb, attn = self.embed(images, view_counts)
        logits = self.head(subject_emb)                        # [B, n_classes]
        return logits, attn


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

@dataclass
class TrainConfig:
    backbone:      str = "resnet34"
    img_size:      int = IMG_SIZE
    max_views:     int = 8
    batch_size:    int = 8          # subjects per batch (~30 images/batch)
    max_epochs:    int = 30
    patience:      int = 7
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    seed:          int = 42          # model init / augmentation / training stochasticity
    split_seed:    int = 42          # subject train/val/test split — must stay FIXED
                                      # across ensemble members so they're trained on the
                                      # same data and are honestly comparable/combinable


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(cfg: TrainConfig = TrainConfig(), save_path: str = ARTIFACT_PATH,
          verbose_prefix: str = "ImagingAgent") -> Dict:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
        balanced_accuracy_score, confusion_matrix
    import pandas as pd
    from agent_utils import calibrate_threshold

    _set_seed(cfg.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    by_subject = build_subject_index()
    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    train_ids, val_ids, test_ids = group_stratified_split(subjects_df, seed=cfg.split_seed)
    train_ids = [s for s in train_ids if s in by_subject]
    val_ids   = [s for s in val_ids if s in by_subject]
    test_ids  = [s for s in test_ids if s in by_subject]

    labels_train = [by_subject[s][0].label for s in train_ids]
    n_pos = sum(labels_train); n_neg = len(labels_train) - n_pos
    class_weights = torch.tensor(
        [len(labels_train) / (2.0 * max(n_neg, 1)), len(labels_train) / (2.0 * max(n_pos, 1))],
        dtype=torch.float32)
    print(f"[{verbose_prefix}] train subjects={len(train_ids)} (healthy={n_neg}, patient={n_pos}) "
          f"| val={len(val_ids)} | test={len(test_ids)}")
    print(f"[{verbose_prefix}] class weights (healthy, patient) = {class_weights.tolist()}")

    train_ds = MultiViewSubjectDataset(train_ids, by_subject, cfg.img_size,
                                        augment=True, max_views=cfg.max_views)
    val_ds = MultiViewSubjectDataset(val_ids, by_subject, cfg.img_size,
                                      augment=False, max_views=cfg.max_views)
    test_ds = MultiViewSubjectDataset(test_ids, by_subject, cfg.img_size,
                                       augment=False, max_views=cfg.max_views)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_subjects, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_subjects, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_subjects, num_workers=0)

    model = ImagingNet(backbone=cfg.backbone, n_classes=2, pretrained=True)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                             factor=0.5, patience=3)

    def run_epoch(loader, train_mode: bool):
        model.train(train_mode)
        total_loss, all_p, all_y = 0.0, [], []
        for images, view_counts, labels, sids in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.set_grad_enabled(train_mode):
                logits, _ = model(images, view_counts)
                loss = criterion(logits, labels)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_p.extend(probs.tolist())
            all_y.extend(labels.cpu().numpy().tolist())
        avg_loss = total_loss / max(len(all_y), 1)
        return avg_loss, np.array(all_p), np.array(all_y)

    best_val_bacc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_p, tr_y = run_epoch(train_loader, train_mode=True)
        va_loss, va_p, va_y = run_epoch(val_loader, train_mode=False)

        epoch_threshold = calibrate_threshold(va_y, va_p) if len(va_y) else 0.5
        va_pred = (va_p >= epoch_threshold).astype(int)
        va_bacc = balanced_accuracy_score(va_y, va_pred) if len(va_y) else 0.0
        va_auc = roc_auc_score(va_y, va_p) if len(set(va_y.tolist())) > 1 else float("nan")
        scheduler.step(va_bacc)

        elapsed = time.time() - t0
        print(f"[{verbose_prefix}] epoch {epoch:02d} | train_loss={tr_loss:.4f} "
              f"| val_loss={va_loss:.4f} val_bacc={va_bacc:.3f} val_auc={va_auc:.3f} "
              f"| {elapsed:.0f}s elapsed")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                         "val_balanced_accuracy": va_bacc, "val_auc": va_auc})

        if va_bacc > best_val_bacc:
            best_val_bacc = va_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[{verbose_prefix}] Early stopping at epoch {epoch} "
                      f"(no val improvement in {cfg.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Recompute val probabilities with the restored best-checkpoint weights
    # and calibrate the deployed decision threshold on them (test stays untouched).
    _, va_p_final, va_y_final = run_epoch(val_loader, train_mode=False)
    threshold = calibrate_threshold(va_y_final, va_p_final) if len(va_y_final) else 0.5

    te_loss, te_p, te_y = run_epoch(test_loader, train_mode=False)
    te_pred = (te_p >= threshold).astype(int)
    metrics = {
        "test_n": len(te_y),
        "test_accuracy": float(accuracy_score(te_y, te_pred)) if len(te_y) else float("nan"),
        "test_balanced_accuracy": float(balanced_accuracy_score(te_y, te_pred)) if len(te_y) else float("nan"),
        "test_f1": float(f1_score(te_y, te_pred, zero_division=0)) if len(te_y) else float("nan"),
        "test_auc": float(roc_auc_score(te_y, te_p)) if len(set(te_y.tolist())) > 1 else float("nan"),
        "best_val_balanced_accuracy": float(best_val_bacc),
        "decision_threshold": threshold,
        "train_time_sec": time.time() - t0,
    }
    if len(te_y):
        cm = confusion_matrix(te_y, te_pred, labels=[0, 1])
        metrics["test_confusion_matrix"] = cm.tolist()

    print(f"[{verbose_prefix}] Final test metrics:")
    for k, v in metrics.items():
        print(f"    {k:28s}: {v}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": cfg.backbone,
                "img_size": cfg.img_size, "max_views": cfg.max_views,
                "threshold": threshold, "seed": cfg.seed,
                "metrics": metrics, "history": history}, save_path)
    print(f"[{verbose_prefix}] Saved -> {save_path}")
    return metrics


# ─────────────────────────────────────────────
# Inference API
# ─────────────────────────────────────────────

@dataclass
class ImagingPrediction:
    subject_id:  str
    available:   bool
    probability: Optional[float] = None
    confidence:  float = 0.0
    view_attention: Optional[List[float]] = None


class ImagingAgent:
    def __init__(self, artifact_path: str = ARTIFACT_PATH,
                 processed_dir: str = PROCESSED_ROOT):
        if not os.path.exists(artifact_path):
            train()
        blob = torch.load(artifact_path, map_location="cpu", weights_only=False)
        self.model = ImagingNet(backbone=blob["backbone"], pretrained=False)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.img_size = blob["img_size"]
        self.max_views = blob["max_views"]
        self.threshold = blob.get("threshold", 0.5)
        self.by_subject = build_subject_index(processed_dir)

    def _subject_tensor(self, subject_id: str) -> Optional[torch.Tensor]:
        if subject_id not in self.by_subject:
            return None
        views = self.by_subject[subject_id]
        if len(views) > self.max_views:
            idx = np.linspace(0, len(views) - 1, self.max_views).round().astype(int)
            views = [views[j] for j in idx]
        tensors = [image_to_tensor(v.image, self.img_size, augment=False) for v in views]
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def predict(self, subject_id: str) -> ImagingPrediction:
        images = self._subject_tensor(subject_id)
        if images is None:
            return ImagingPrediction(subject_id=subject_id, available=False)
        logits, attn = self.model(images, [images.shape[0]])
        p = float(torch.softmax(logits, dim=1)[0, 1].item())
        from agent_utils import signed_margin
        return ImagingPrediction(
            subject_id=subject_id, available=True, probability=p,
            confidence=abs(signed_margin(p, self.threshold)),
            view_attention=attn[0].tolist())

    @torch.no_grad()
    def embed(self, subject_id: str) -> Optional[np.ndarray]:
        """Subject-level 512-d embedding, for OOD / representation analysis."""
        images = self._subject_tensor(subject_id)
        if images is None:
            return None
        emb, _ = self.model.embed(images, [images.shape[0]])
        return emb[0].numpy()

    def mc_dropout_probs(self, subject_id: str, T: int = 30) -> Optional[np.ndarray]:
        """T stochastic forward passes with dropout left active — the same
        subject's views repeated T times in one batched call so each pass
        draws an independent dropout mask."""
        images = self._subject_tensor(subject_id)
        if images is None:
            return None
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        n_views = images.shape[0]
        images_rep = images.repeat(T, 1, 1, 1)
        with torch.no_grad():
            logits, _ = self.model(images_rep, [n_views] * T)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        self.model.eval()   # restore fully-deterministic mode
        return probs

    def predict_batch(self, subject_ids: List[str]):
        import pandas as pd
        rows = [self.predict(sid).__dict__ for sid in subject_ids]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Deep Ensemble (Phase 3, uncertainty_agent.py's third uncertainty source)
# ─────────────────────────────────────────────

ENSEMBLE_DIR = r"C:/PhD/Implementation/artifacts/ensemble"
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]   # M=5 independent initializations


def train_ensemble(seeds: List[int] = ENSEMBLE_SEEDS, split_seed: int = 42) -> List[Dict]:
    """
    Train M independently-initialized ImagingNet models on the SAME
    train/val/test split (split_seed fixed) — only weight init, batch
    shuffling and augmentation draws differ between members. Deep Ensembles'
    uncertainty signal is inter-model disagreement, which only means
    something if every member saw the same data.
    """
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    all_metrics = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}\n[DeepEnsemble] Training member {i+1}/{len(seeds)} (seed={seed})\n{'='*60}")
        cfg = TrainConfig(seed=seed, split_seed=split_seed)
        save_path = f"{ENSEMBLE_DIR}/member_{i}_seed{seed}.pt"
        metrics = train(cfg, save_path=save_path, verbose_prefix=f"DeepEnsemble/m{i}")
        all_metrics.append({"member": i, "seed": seed, **metrics})
    return all_metrics


class ImagingEnsemble:
    """Loads all trained ensemble members for MC-style disagreement scoring."""

    def __init__(self, ensemble_dir: str = ENSEMBLE_DIR, processed_dir: str = PROCESSED_ROOT):
        import glob
        paths = sorted(glob.glob(f"{ensemble_dir}/member_*.pt"))
        if not paths:
            raise FileNotFoundError(
                f"No ensemble checkpoints in {ensemble_dir} — run train_ensemble() first.")
        self.models, self.thresholds = [], []
        for p in paths:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            m = ImagingNet(backbone=blob["backbone"], pretrained=False)
            m.load_state_dict(blob["state_dict"])
            m.eval()
            self.models.append(m)
            self.thresholds.append(blob.get("threshold", 0.5))
        self.img_size = torch.load(paths[0], map_location="cpu", weights_only=False)["img_size"]
        self.max_views = torch.load(paths[0], map_location="cpu", weights_only=False)["max_views"]
        self.by_subject = build_subject_index(processed_dir)
        print(f"[ImagingEnsemble] Loaded {len(self.models)} members from {ensemble_dir}")

    @torch.no_grad()
    def predict_all(self, subject_id: str) -> Optional[np.ndarray]:
        """Returns an array of M per-member P(patient), or None if unavailable."""
        if subject_id not in self.by_subject:
            return None
        views = self.by_subject[subject_id]
        if len(views) > self.max_views:
            idx = np.linspace(0, len(views) - 1, self.max_views).round().astype(int)
            views = [views[j] for j in idx]
        tensors = [image_to_tensor(v.image, self.img_size, augment=False) for v in views]
        images = torch.stack(tensors, dim=0)
        probs = []
        for m in self.models:
            logits, _ = m(images, [len(tensors)])
            probs.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
        return np.array(probs)


if __name__ == "__main__":
    train()
