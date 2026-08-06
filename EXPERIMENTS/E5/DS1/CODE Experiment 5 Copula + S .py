"""
EXPERIMENT 5 — GaussianCopula with Data-Derived Personas

For each persona cluster, fit a Gaussian copula on the within-cluster
real data, then sample synthetic responses proportionally to cluster sizes.

The Gaussian copula is implemented from scratch (no SDV dependency):
  1. Per column, compute the empirical CDF and map each value to a standard-
     normal variable via the inverse normal CDF.
  2. Estimate the correlation matrix of the resulting Gaussian variables.
  3. To sample: draw from multivariate normal with that correlation, push back
     through the standard-normal CDF, then through the inverse empirical CDF
     of each column to recover original category labels.
"""
import pandas as pd
import numpy as np
import json
import os
from scipy.stats import norm, rankdata
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = "/home/claude/personas"
np.random.seed(42)

real = pd.read_excel(f"{OUTPUT_DIR}/real_data_with_cluster.xlsx")
with open(f"{OUTPUT_DIR}/cluster_shares.json") as f:
    shares = json.load(f)

feature_cols = [c for c in real.columns if c != "cluster"]
N_SYNTH = 274


def fit_gaussian_copula(df):
    """Fit a Gaussian copula on a DataFrame of categorical/ordinal columns."""
    encoders = {}
    enc_df = pd.DataFrame()
    for col in df.columns:
        le = LabelEncoder()
        enc_df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    n = len(enc_df)
    z = pd.DataFrame()
    for col in enc_df.columns:
        # Smoothed empirical CDF: (rank - 0.5) / n, clipped to avoid ±inf
        ranks = rankdata(enc_df[col].values, method="average")
        u = (ranks - 0.5) / n
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z[col] = norm.ppf(u)

    # Correlation matrix with small regularisation for positive-definiteness
    corr = z.corr().values
    corr = corr + np.eye(len(corr)) * 1e-6

    # Empirical category distributions for inverse CDF mapping at sample time
    empirical_cdf = {}
    for col in enc_df.columns:
        vc = enc_df[col].value_counts(normalize=True).sort_index()
        empirical_cdf[col] = vc

    return {
        "encoders":       encoders,
        "corr":           corr,
        "columns":        list(enc_df.columns),
        "empirical_cdf":  empirical_cdf,
        "n_train":        n,
    }


def sample_gaussian_copula(model, n_samples):
    """Sample synthetic rows from a fitted Gaussian copula."""
    samples = np.random.multivariate_normal(
        mean=np.zeros(len(model["columns"])),
        cov=model["corr"],
        size=n_samples,
    )
    result = pd.DataFrame()
    for i, col in enumerate(model["columns"]):
        # Push through standard-normal CDF to get uniform variates
        u = norm.cdf(samples[:, i])
        # Invert the column's empirical CDF: find smallest cat code k with cumsum[k] >= u
        cdf_table = model["empirical_cdf"][col].sort_index()
        cum = cdf_table.cumsum().values
        codes = cdf_table.index.values

        chosen_codes = []
        for ui in u:
            # number of category buckets strictly below ui
            k = int(np.searchsorted(cum, ui, side="left"))
            k = min(k, len(codes) - 1)
            chosen_codes.append(codes[k])

        # Decode back to original labels
        result[col] = model["encoders"][col].inverse_transform(chosen_codes)
    return result


# ----------------------------------------------------------------------
# Fit one copula per cluster, sample proportionally
# ----------------------------------------------------------------------
clusters = sorted(real["cluster"].unique())
cluster_models = {}
print("Fitting one Gaussian copula per persona cluster...")
for c in clusters:
    sub = real[real["cluster"] == c][feature_cols]
    cluster_models[c] = fit_gaussian_copula(sub)
    print(f"  Cluster {c}: fitted on n={len(sub)} records, "
          f"correlation matrix shape={cluster_models[c]['corr'].shape}")

# Allocate sample sizes per cluster proportionally
n_per_cluster = {}
remaining = N_SYNTH
for c in clusters[:-1]:
    n_per_cluster[c] = int(round(shares[str(c)] * N_SYNTH))
    remaining -= n_per_cluster[c]
n_per_cluster[clusters[-1]] = remaining

print(f"\nSamples per cluster: {n_per_cluster}  (total = {sum(n_per_cluster.values())})")

# Sample
pieces = []
for c, n in n_per_cluster.items():
    sub_synth = sample_gaussian_copula(cluster_models[c], n)
    sub_synth["__cluster"] = c
    pieces.append(sub_synth)

synth_exp5 = pd.concat(pieces, ignore_index=True)
synth_exp5 = synth_exp5.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
synth_exp5 = synth_exp5.drop(columns=["__cluster"])

out_path = f"{OUTPUT_DIR}/exp5_copula_with_personas.xlsx"
synth_exp5.to_excel(out_path, index=False)
print(f"\nExperiment 5 — Copula with personas: saved {len(synth_exp5)} responses to {out_path}")
print(f"\nValue counts for Q1 (Age Group):")
print(synth_exp5["Q1. Age Group"].value_counts())
print(f"\nValue counts for Q11 (Plan to Shop Again):")
print(synth_exp5["Q11. Plan to Shop Again in Future"].value_counts())
