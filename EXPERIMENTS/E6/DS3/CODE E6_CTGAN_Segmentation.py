"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 6 — CTGAN + SEGMENTATION                                     ║
║  Dataset: Sustainable Apparel Consumer Behaviour Survey (34 columns)    ║
║  Library: SDV (Synthetic Data Vault)                                     ║
║  Method: CTGANSynthesizer fitted independently per K-Means segment       ║
╚══════════════════════════════════════════════════════════════════════════╝

Install: pip install sdv openpyxl

A reduced batch size (38) is used to accommodate the smallest segment
(n≈38 for Segment 3 — "Eco-Pragmatic Buyers").
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

SEED = 42
np.random.seed(SEED)
N_SYNTH = 384

real = pd.read_excel("REAL_DATA_APPAREL.xlsx")
cols = list(real.columns)
print(f"Real data: {real.shape}")

km = KMeans(n_clusters=5, random_state=SEED, n_init=50)
labels = km.fit_predict(real[cols].values)
real["cluster"] = labels

shares = {c:(labels==c).sum()/len(labels) for c in range(5)}
n_per  = {}; rem = N_SYNTH
for c in range(4):
    n_per[c] = int(round(shares[c]*N_SYNTH)); rem -= n_per[c]
n_per[4] = rem
print("Segment n:", n_per)

# ── FIT CTGAN PER SEGMENT ────────────────────────────────────────────────────
pieces = []
for c in range(5):
    sub = real[real["cluster"] == c][cols].reset_index(drop=True)
    print(f"\nSegment {c}: fitting CTGAN on n={len(sub)}...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(sub)
    for col in sub.columns:
        metadata.update_column(column_name=col, sdtype="categorical")

    syn = CTGANSynthesizer(
        metadata,
        epochs=500,
        batch_size=38,   # small batch for small cluster sizes (Segment 3 n≈38)
        verbose=False,
    )
    syn.fit(sub)
    chunk = syn.sample(num_rows=n_per[c])
    pieces.append(chunk)
    print(f"   Sampled {n_per[c]} rows.")

synth = (pd.concat(pieces, ignore_index=True)
           .sample(frac=1, random_state=SEED)
           .reset_index(drop=True))

synth.to_excel("exp6_ctgan_segmentation.xlsx", index=False)
print(f"\nSaved: exp6_ctgan_segmentation.xlsx  ({len(synth)} rows)")
print("\nPI2 (target) distribution:")
print(synth["PI2"].value_counts().sort_index())
