"""
Phase 3: Uncertainty Quantification Pipeline.

Three complementary uncertainty sources, matching the diagram:

  Bayesian Neural Network : a genuine variational-inference (Bayes-by-
      Backprop) MLP — not MC-Dropout relabeled — trained on the Texture
      Agent's own standardized+selected 40-feature representation (reusing
      its fitted StandardScaler + SelectKBest exactly, so the BNN sees the
      same trusted feature space the Texture Agent's GradientBoosting model
      does). Every weight is a Gaussian posterior; sampling the posterior T
      times gives an epistemic-uncertainty estimate.

  Monte Carlo Dropout : T=30 stochastic forward passes through the Imaging
      Agent with its head's Dropout(0.3) left active at inference — a
      subject's views are repeated T times in ONE batched forward call so
      each pass draws an independent dropout mask without T separate
      Python-level calls.

  Deep Ensembles : disagreement across the M=5 independently-initialized
      ImagingNet models trained by imaging_agent.train_ensemble().

Uncertainty Aggregation Layer combines the three into one calibrated score,
plus Mahalanobis-distance OOD detection on the Imaging Agent's embedding
space, and routes into the diagram's three tiers: Low -> classify,
Medium -> triage (~20%), High/OOD -> reject/escalate.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from data_prep import SUBJECT_TABLE_CSV, group_stratified_split
from imaging_agent import ImagingAgent, ImagingEnsemble, ENSEMBLE_DIR
from texture_classifier import SUBJECT_FEATURES_CSV, ARTIFACT_PATH as TEXTURE_ARTIFACT_PATH

BNN_ARTIFACT = r"C:/PhD/Implementation/artifacts/bnn_agent.pt"
OOD_ARTIFACT = r"C:/PhD/Implementation/artifacts/ood_stats.joblib"
MC_DROPOUT_T = 30
BNN_POSTERIOR_T = 30


# ─────────────────────────────────────────────
# Bayesian Neural Network — Bayes-by-Backprop
# ─────────────────────────────────────────────

class BayesianLinear(nn.Module):
    """A linear layer with a fully-factorized Gaussian posterior over every
    weight and bias (mean-field variational inference), reparameterized so
    gradients flow through the sampled weights (Blundell et al., 2015)."""

    def __init__(self, in_f: int, out_f: int, prior_std: float = 1.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_f, in_f).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))
        self.prior_std = prior_std

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w_std = F.softplus(self.w_rho)
            b_std = F.softplus(self.b_rho)
            w = self.w_mu + w_std * torch.randn_like(w_std)
            b = self.b_mu + b_std * torch.randn_like(b_std)
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        def kl_gaussian(mu, std):
            var, prior_var = std ** 2, self.prior_std ** 2
            return 0.5 * torch.sum(var / prior_var + mu ** 2 / prior_var - 1 - torch.log(var / prior_var))
        return kl_gaussian(self.w_mu, F.softplus(self.w_rho)) + \
               kl_gaussian(self.b_mu, F.softplus(self.b_rho))


class BayesianMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden)
        self.fc2 = BayesianLinear(hidden, n_classes)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = F.relu(self.fc1(x, sample=sample))
        return self.fc2(h, sample=sample)

    def kl(self) -> torch.Tensor:
        return self.fc1.kl() + self.fc2.kl()


def _texture_bnn_features() -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Reproduces the Texture Agent's own standardized+selected 40-d input
    space exactly, from its fitted pipeline — the BNN trains on the same
    trusted representation, not a fresh/uncontrolled feature set."""
    blob = joblib.load(TEXTURE_ARTIFACT_PATH)
    pipe, feat_cols = blob["pipeline"], blob["feature_names"]
    subj = pd.read_csv(SUBJECT_FEATURES_CSV)
    X_full = subj[feat_cols].to_numpy(dtype=np.float64)
    X_scaled = pipe.named_steps["scale"].transform(X_full)
    mask = pipe.named_steps["select"].get_support()
    X_sel = X_scaled[:, mask]
    selected_names = [f for f, m in zip(feat_cols, mask) if m]
    return X_sel, selected_names, subj[["subject_id", "label"]]


def train_bnn(seed: int = 42, epochs: int = 400, lr: float = 5e-3) -> Dict:
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
    from agent_utils import calibrate_threshold, balanced_sample_weights

    torch.manual_seed(seed)
    X, feat_names, meta = _texture_bnn_features()
    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    train_ids, val_ids, test_ids = group_stratified_split(subjects_df, seed=seed)

    idx = {sid: i for i, sid in enumerate(meta["subject_id"])}
    def subset(ids):
        rows = [idx[s] for s in ids if s in idx]
        return X[rows], meta["label"].to_numpy()[rows]

    X_tr, y_tr = subset(train_ids)
    X_va, y_va = subset(val_ids)
    X_te, y_te = subset(test_ids)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    sw = torch.tensor(balanced_sample_weights(y_tr), dtype=torch.float32)

    model = BayesianMLP(in_dim=X.shape[1], hidden=32, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = len(y_tr)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt, sample=True)
        nll = F.cross_entropy(logits, yt, reduction="none")
        nll = (nll * sw).mean()
        loss = nll + model.kl() / n_train
        loss.backward()
        opt.step()

    # Posterior-predictive on val (T samples, averaged) for threshold calibration.
    def posterior_predict(Xnp: np.ndarray, T: int) -> np.ndarray:
        model.eval()
        Xin = torch.tensor(Xnp, dtype=torch.float32)
        samples = []
        with torch.no_grad():
            for _ in range(T):
                logits = model(Xin, sample=True)
                samples.append(torch.softmax(logits, dim=1)[:, 1].numpy())
        return np.stack(samples, axis=0)   # (T, N)

    va_samples = posterior_predict(X_va, BNN_POSTERIOR_T) if len(y_va) else np.zeros((BNN_POSTERIOR_T, 0))
    va_mean = va_samples.mean(axis=0)
    threshold = calibrate_threshold(y_va, va_mean) if len(y_va) else 0.5

    te_samples = posterior_predict(X_te, BNN_POSTERIOR_T) if len(y_te) else np.zeros((BNN_POSTERIOR_T, 0))
    te_mean = te_samples.mean(axis=0)
    te_pred = (te_mean >= threshold).astype(int)
    metrics = {
        "test_n": len(y_te),
        "test_accuracy": float(accuracy_score(y_te, te_pred)) if len(y_te) else float("nan"),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, te_pred)) if len(y_te) else float("nan"),
        "test_auc": float(roc_auc_score(y_te, te_mean)) if len(set(y_te.tolist())) > 1 else float("nan"),
        "decision_threshold": threshold,
        "mean_posterior_std": float(te_samples.std(axis=0).mean()) if te_samples.size else float("nan"),
    }

    os.makedirs(os.path.dirname(BNN_ARTIFACT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1],
                "feat_names": feat_names, "threshold": threshold, "metrics": metrics}, BNN_ARTIFACT)
    print("[BNN] Bayes-by-Backprop trained on Texture Agent's 40-d selected feature space:")
    for k, v in metrics.items():
        print(f"    {k:24s}: {v}")
    print(f"[BNN] Saved -> {BNN_ARTIFACT}")
    return metrics


class BNNAgent:
    def __init__(self, artifact_path: str = BNN_ARTIFACT):
        if not os.path.exists(artifact_path):
            train_bnn()
        blob = torch.load(artifact_path, map_location="cpu", weights_only=False)
        self.model = BayesianMLP(in_dim=blob["in_dim"])
        self.model.load_state_dict(blob["state_dict"])
        self.feat_names = blob["feat_names"]
        self.threshold = blob["threshold"]
        X, feat_names, meta = _texture_bnn_features()
        assert feat_names == self.feat_names, "Texture Agent's selected features changed — retrain the BNN."
        self._X = pd.DataFrame(X, index=meta["subject_id"])

    def posterior_samples(self, subject_id: str, T: int = BNN_POSTERIOR_T) -> Optional[np.ndarray]:
        if subject_id not in self._X.index:
            return None
        x = torch.tensor(self._X.loc[[subject_id]].to_numpy(), dtype=torch.float32)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for _ in range(T):
                logits = self.model(x, sample=True)
                probs.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
        return np.array(probs)


# ─────────────────────────────────────────────
# Out-of-distribution detection — Mahalanobis distance
# ─────────────────────────────────────────────

OOD_PCA_COMPONENTS = 20


def fit_ood_stats(seed: int = 42) -> Dict:
    """
    Class-conditional mean + shared covariance of the Imaging Agent's
    subject embedding, fit on TRAIN subjects only — but first projected
    from 512-d down to OOD_PCA_COMPONENTS via PCA (also fit on train).

    With only 161 train subjects, estimating a 512x512 covariance is
    badly underdetermined (a train-calibrated 97.5th-percentile threshold
    flagged ~66% of held-out IN-distribution test subjects as OOD before
    this fix — a textbook symptom of an unstable high-dimensional
    covariance). Reducing to a ~20-d PCA subspace first is the standard
    remedy: it keeps the sample-to-dimension ratio in a regime where the
    covariance estimate — and therefore the OOD threshold — actually
    generalizes to held-out data.
    """
    from sklearn.decomposition import PCA

    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    train_ids, _, _ = group_stratified_split(subjects_df, seed=seed)
    label_map = subjects_df.set_index("subject_id")["label"].to_dict()

    agent = ImagingAgent()
    embs, labels = [], []
    for sid in train_ids:
        e = agent.embed(sid)
        if e is not None:
            embs.append(e)
            labels.append(label_map[sid])
    embs = np.stack(embs)
    labels = np.array(labels)

    n_components = min(OOD_PCA_COMPONENTS, embs.shape[0] - 1, embs.shape[1])
    pca = PCA(n_components=n_components, random_state=seed).fit(embs)
    Z = pca.transform(embs)

    means = {c: Z[labels == c].mean(axis=0) for c in np.unique(labels)}
    centered = np.concatenate([Z[labels == c] - means[c] for c in np.unique(labels)], axis=0)
    cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(n_components)
    inv_cov = np.linalg.inv(cov)

    def mahalanobis(z, mean):
        d = z - mean
        return float(np.sqrt(d @ inv_cov @ d))

    train_dists = np.array([
        min(mahalanobis(z, means[c]) for c in means) for z in Z])
    ood_threshold = float(np.percentile(train_dists, 97.5))

    blob = {"pca": pca, "means": means, "inv_cov": inv_cov, "ood_threshold": ood_threshold,
            "train_dist_mean": float(train_dists.mean()), "train_dist_std": float(train_dists.std())}
    os.makedirs(os.path.dirname(OOD_ARTIFACT), exist_ok=True)
    joblib.dump(blob, OOD_ARTIFACT)
    print(f"[OOD] Fit on {len(embs)} train embeddings, PCA -> {n_components}d. "
          f"threshold(97.5th pct)={ood_threshold:.2f}")
    return {"n_train": len(embs), "pca_components": n_components, "ood_threshold": ood_threshold}


class OODDetector:
    def __init__(self, artifact_path: str = OOD_ARTIFACT):
        if not os.path.exists(artifact_path):
            fit_ood_stats()
        blob = joblib.load(artifact_path)
        self.pca = blob["pca"]
        self.means = blob["means"]
        self.inv_cov = blob["inv_cov"]
        self.threshold = blob["ood_threshold"]

    def score(self, embedding: np.ndarray) -> Tuple[float, bool]:
        z = self.pca.transform(embedding.reshape(1, -1))[0]
        dists = []
        for c, mean in self.means.items():
            d = z - mean
            dists.append(float(np.sqrt(d @ self.inv_cov @ d)))
        min_dist = min(dists)
        return min_dist, bool(min_dist > self.threshold)


# ─────────────────────────────────────────────
# Uncertainty Aggregation Layer
# ─────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    subject_id: str
    p_mean: float                 # aggregated P(patient) across all sources that fired
    u_bayesian: Optional[float]   # BNN posterior variance
    u_mc: Optional[float]         # MC-Dropout variance
    u_ensemble: Optional[float]   # Deep Ensemble variance
    u_combined: float             # aggregated uncertainty in [0, ~0.25]
    ood_distance: Optional[float]
    is_ood: bool
    tier: str                     # "low" | "medium" | "high_ood"


class UncertaintyAgent:
    def __init__(self):
        self.imaging = ImagingAgent()
        self.bnn = BNNAgent()
        self.ood = OODDetector()
        try:
            self.ensemble = ImagingEnsemble()
        except FileNotFoundError:
            self.ensemble = None
            print("[UncertaintyAgent] No deep-ensemble checkpoints found yet — "
                  "u_ensemble will be omitted until imaging_agent.train_ensemble() runs.")

    def assess(self, subject_id: str) -> UncertaintyResult:
        sources_p, sources_u = [], {}

        mc = self.imaging.mc_dropout_probs(subject_id, T=MC_DROPOUT_T)
        if mc is not None:
            sources_p.append(mc.mean())
            sources_u["u_mc"] = float(mc.var())

        bnn_samples = self.bnn.posterior_samples(subject_id)
        if bnn_samples is not None:
            sources_p.append(bnn_samples.mean())
            sources_u["u_bayesian"] = float(bnn_samples.var())

        if self.ensemble is not None:
            ens = self.ensemble.predict_all(subject_id)
            if ens is not None:
                sources_p.append(ens.mean())
                sources_u["u_ensemble"] = float(ens.var())

        p_mean = float(np.mean(sources_p)) if sources_p else 0.5
        u_combined = float(np.mean(list(sources_u.values()))) if sources_u else 1.0

        emb = self.imaging.embed(subject_id)
        ood_dist, is_ood = (None, False)
        if emb is not None:
            ood_dist, is_ood = self.ood.score(emb)

        if is_ood:
            tier = "high_ood"
        elif u_combined <= 0.02:
            tier = "low"
        else:
            tier = "medium"

        return UncertaintyResult(
            subject_id=subject_id, p_mean=p_mean,
            u_bayesian=sources_u.get("u_bayesian"), u_mc=sources_u.get("u_mc"),
            u_ensemble=sources_u.get("u_ensemble"), u_combined=u_combined,
            ood_distance=ood_dist, is_ood=is_ood, tier=tier)

    def assess_batch(self, subject_ids: List[str]) -> pd.DataFrame:
        from dataclasses import asdict
        return pd.DataFrame([asdict(self.assess(sid)) for sid in subject_ids])


# ─────────────────────────────────────────────
# Evaluation: does higher uncertainty actually predict more error?
# ─────────────────────────────────────────────

def evaluate(seed: int = 42) -> Dict:
    subjects_df = pd.read_csv(SUBJECT_TABLE_CSV)
    label_map = subjects_df.set_index("subject_id")["label"].to_dict()
    _, _, test_ids = group_stratified_split(subjects_df, seed=seed)

    agent = UncertaintyAgent()
    results = agent.assess_batch(test_ids)
    results["true_label"] = results["subject_id"].map(label_map)
    results["pred_label"] = (results["p_mean"] >= 0.5).astype(int)
    results["correct"] = (results["pred_label"] == results["true_label"]).astype(int)

    report: Dict = {"n_test_subjects": len(results)}
    for tier in ("low", "medium", "high_ood"):
        sub = results[results["tier"] == tier]
        report[f"tier_{tier}_n"] = len(sub)
        report[f"tier_{tier}_pct"] = round(100 * len(sub) / max(len(results), 1), 1)
        report[f"tier_{tier}_accuracy"] = float(sub["correct"].mean()) if len(sub) else float("nan")

    # Does u_combined correlate with actual mistakes? (higher = better calibrated uncertainty)
    if results["u_combined"].nunique() > 1:
        report["uncertainty_error_correlation"] = float(
            np.corrcoef(results["u_combined"], 1 - results["correct"])[0, 1])

    print("[UncertaintyAgent] Test-split uncertainty report:")
    for k, v in report.items():
        print(f"    {k:32s}: {v}")
    results.to_csv(r"C:/PhD/Implementation/artifacts/uncertainty_test_results.csv", index=False)
    return report


if __name__ == "__main__":
    fit_ood_stats()
    train_bnn()
    evaluate()
