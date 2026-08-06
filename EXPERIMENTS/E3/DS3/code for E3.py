"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 3 — GAUSSIAN COPULA (Global)                                 ║
║  Dataset: Sustainable Apparel Consumer Behaviour Survey (34 columns)    ║
║  Library: SDV (Synthetic Data Vault)                                     ║
║  Method: Gaussian copula · no segmentation · ordinal-ordering dict       ║
╚══════════════════════════════════════════════════════════════════════════╝

Install: pip install sdv openpyxl
"""

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

REAL_FILE = "REAL_DATA_APPAREL.xlsx"
N_SYNTH   = 384

real = pd.read_excel(REAL_FILE)
print(f"Real data: {real.shape}")

# ── ORDINAL ORDERING DICTIONARY (all columns are ordinal 1..N) ─────────────
ORDINAL_ORDER = {
    "gender":    [1, 2],
    "age":       [1, 2, 3, 4],
    "education": [1, 2, 3, 4],
    "occupation":[1, 2, 3, 4],
    "city":      [1, 2, 3, 4, 5],
    "income":    [1, 2, 3],
}
LIKERT_ITEMS = [c for c in real.columns if c not in ORDINAL_ORDER]
for c in LIKERT_ITEMS:
    ORDINAL_ORDER[c] = [1, 2, 3, 4, 5]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)
for col in real.columns:
    metadata.update_column(column_name=col, sdtype="categorical", order=ORDINAL_ORDER[col])
print("Metadata with ordinal ordering built (34 columns).")

synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution="gaussian_kde")
synthesizer.fit(real)
print("GaussianCopula fitted on real data.")

synth = synthesizer.sample(num_rows=N_SYNTH)
print(f"Generated: {synth.shape}")

synth.to_excel("exp3_copula_global.xlsx", index=False)
print("Saved: exp3_copula_global.xlsx")

print("\nPI2 (target) distribution:")
print(synth["PI2"].value_counts().sort_index())
