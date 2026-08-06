"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 5 — GAUSSIAN COPULA + SEGMENTATION                           ║
║  Dataset: Sustainable Apparel Consumer Behaviour Survey (34 columns)    ║
║  Method: From-scratch Gaussian copula · one per K-Means segment          ║
║  No external library beyond numpy / scipy / sklearn                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.cluster import KMeans

SEED = 42
np.random.seed(SEED)
N_SYNTH = 384

real = pd.read_excel("REAL_DATA_APPAREL.xlsx")
cols = list(real.columns)
print(f"Real data: {real.shape}")

COL_RANGES = {c: (int(real[c].min()), int(real[c].max())) for c in cols}

# ── K-MEANS K=5 ───────────────────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=SEED, n_init=50)
labels = km.fit_predict(real[cols].values)
real["cluster"] = labels

shares = {c:(labels==c).sum()/len(labels) for c in range(5)}
n_per  = {}; rem = N_SYNTH
for c in range(4):
    n_per[c] = int(round(shares[c]*N_SYNTH)); rem -= n_per[c]
n_per[4] = rem
print("Segment n:", n_per)

# ── GAUSSIAN COPULA HELPERS ───────────────────────────────────────────────
def make_psd(M, reg=1e-4):
    M = (M + M.T) / 2
    M = np.nan_to_num(M)
    e, v = np.linalg.eigh(M)
    return v @ np.diag(np.maximum(e, reg)) @ v.T

def fit_copula(sub):
    n = len(sub)
    Z, ecdf = pd.DataFrame(), {}
    for col in sub.columns:
        ranks = rankdata(sub[col].values, method="average")
        u = np.clip((ranks - 0.5) / n, 1e-6, 1 - 1e-6)
        Z[col] = norm.ppf(u)
        ecdf[col] = sub[col].value_counts(normalize=True).sort_index()
    corr = np.nan_to_num(np.corrcoef(Z.values.T))
    corr = make_psd(corr)
    return {"corr": corr, "ecdf": ecdf, "columns": list(sub.columns)}

def sample_copula(model, n_samples):
    try:
        z = np.random.multivariate_normal(
            np.zeros(len(model["columns"])), model["corr"], size=n_samples)
    except np.linalg.LinAlgError:
        z = np.random.randn(n_samples, len(model["columns"]))
    result = pd.DataFrame()
    for i, col in enumerate(model["columns"]):
        u = norm.cdf(z[:, i])
        ct = model["ecdf"][col].sort_index()
        cum, cats = ct.cumsum().values, ct.index.values
        chosen = [cats[min(np.searchsorted(cum, ui, side="left"),
                           len(cats) - 1)] for ui in u]
        result[col] = chosen
    return result

def clip_round(df, cols):
    out = pd.DataFrame()
    for c in cols:
        lo, hi = COL_RANGES[c]
        out[c] = np.clip(np.round(df[c].values).astype(int), lo, hi)
    return out

# ── FIT ONE COPULA PER SEGMENT & SAMPLE ───────────────────────────────────
pieces = []
for c in range(5):
    sub = real[real["cluster"] == c][cols]
    model = fit_copula(sub)
    chunk = sample_copula(model, n_per[c])
    pieces.append(clip_round(chunk, cols))
    print(f"Segment {c}: sampled {n_per[c]} rows.")

synth = (pd.concat(pieces, ignore_index=True)
           .sample(frac=1, random_state=SEED)
           .reset_index(drop=True))

print(f"\nGenerated: {synth.shape}")
synth.to_excel("exp5_copula_segmentation.xlsx", index=False)
print("Saved: exp5_copula_segmentation.xlsx")
print("\nPI2 (target) distribution:")
print(synth["PI2"].value_counts().sort_index())
