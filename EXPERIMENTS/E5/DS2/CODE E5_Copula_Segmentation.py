"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 5 — GAUSSIAN COPULA + SEGMENTATION                           ║
║  Dataset: Vietnam Online Privacy Survey                                  ║
║  Method: From-scratch Gaussian copula · one per K-Means segment          ║
║  No external library beyond numpy / scipy / sklearn                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
N_SYNTH = 467

# ── LOAD REAL DATA ────────────────────────────────────────────────────────
real = pd.read_excel("REAL_DATA_PRIVACY.xlsx")
cols = list(real.columns)
print(f"Real data: {real.shape}")

# ── ENCODE ────────────────────────────────────────────────────────────────
encoders = {}
real_enc = pd.DataFrame()
for col in cols:
    le = LabelEncoder()
    le.fit(real[col].astype(str))
    encoders[col] = le
    real_enc[col] = le.transform(real[col].astype(str))

# ── K-MEANS K=5 ───────────────────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=SEED, n_init=50)
labels = km.fit_predict(real_enc.values)
real["cluster"] = labels

shares = {c:(labels==c).sum()/len(labels) for c in range(5)}
n_per  = {}; rem = N_SYNTH
for c in range(4):
    n_per[c] = int(round(shares[c]*N_SYNTH)); rem -= n_per[c]
n_per[4] = rem
print("Segment n:", n_per)


# ── GAUSSIAN COPULA HELPERS ───────────────────────────────────────────────
def make_psd(M, reg=1e-4):
    """Ensure matrix is positive semi-definite via eigenvalue clamping."""
    M = (M + M.T) / 2
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

def decode_df(df_enc):
    out = pd.DataFrame()
    for col in cols:
        vals = np.round(df_enc[col].values).astype(int)
        vals = np.clip(vals, 0, len(encoders[col].classes_) - 1)
        out[col] = encoders[col].inverse_transform(vals)
    return out


# ── FIT ONE COPULA PER SEGMENT & SAMPLE ───────────────────────────────────
pieces = []
for c in range(5):
    sub = real_enc[real["cluster"] == c][cols]
    model = fit_copula(sub)
    chunk = sample_copula(model, n_per[c])
    out = pd.DataFrame()
    for col in cols:
        vals = np.clip(chunk[col].values.astype(int),
                       0, len(encoders[col].classes_) - 1)
        out[col] = encoders[col].inverse_transform(vals)
    pieces.append(out)
    print(f"Segment {c}: sampled {n_per[c]} rows.")

synth = (pd.concat(pieces, ignore_index=True)
           .sample(frac=1, random_state=SEED)
           .reset_index(drop=True))

print(f"\nGenerated: {synth.shape}")
synth.to_excel("exp5_copula_segmentation.xlsx", index=False)
print("Saved: exp5_copula_segmentation.xlsx")
print("\nIB4 distribution:")
print(synth["IB4"].value_counts().sort_index())
