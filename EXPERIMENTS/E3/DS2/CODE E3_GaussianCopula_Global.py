"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 3 — GAUSSIAN COPULA (Global)                                 ║
║  Dataset: Vietnam Online Privacy Survey                                  ║
║  Library: SDV (Synthetic Data Vault)                                     ║
║  Method: Gaussian copula · no segmentation · ordinal-ordering dict       ║
╚══════════════════════════════════════════════════════════════════════════╝

Install: pip install sdv openpyxl
"""

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# ── LOAD REAL DATA ────────────────────────────────────────────────────────
REAL_FILE = "REAL_DATA_PRIVACY.xlsx"
N_SYNTH   = 467

real = pd.read_excel(REAL_FILE)
print(f"Real data: {real.shape}")

# ── ORDINAL ORDERING DICTIONARY ───────────────────────────────────────────
# Tells SDV the correct order of categories for each ordinal/Likert scale
ORDINAL_ORDER = {
    # Shopping frequency
    "q4": [2, 3, 4, 5],
    # Privacy concern (ordinal severity)
    "q5": [1, 2, 3, 4, 5],
    # All Likert scales 1–5
    "COL1": [1, 2, 3, 4, 5],
    "COL2": [1, 2, 3, 4, 5],
    "COL3": [1, 2, 3, 4, 5],
    "COL4": [1, 2, 3, 4, 5],
    "ERR1": [1, 2, 3, 4, 5],
    "ERR2": [1, 2, 3, 4, 5],
    "ERR3": [1, 2, 3, 4, 5],
    "ERR4": [1, 2, 3, 4, 5],
    "USU1": [1, 2, 3, 4, 5],
    "USU2": [1, 2, 3, 4, 5],
    "USU3": [1, 2, 3, 4, 5],
    "USU4": [1, 2, 3, 4, 5],
    "IMA1": [1, 2, 3, 4, 5],
    "IMA2": [1, 2, 3, 4, 5],
    "IMA3": [1, 2, 3, 4, 5],
    "OT1":  [1, 2, 3, 4, 5],
    "OT2":  [1, 2, 3, 4, 5],
    "OT3":  [1, 2, 3, 4, 5],
    "OT4":  [1, 2, 3, 4, 5],
    "IB1":  [1, 2, 3, 4, 5],
    "IB2":  [1, 2, 3, 4, 5],
    "IB3":  [1, 2, 3, 4, 5],
    "IB4":  [1, 2, 3, 4, 5],
    "age":  [1, 2, 3, 4],
    "occupation": [1, 2, 3, 4, 5],
}

# Nominal columns (no ordering)
NOMINAL = ["q1", "q2", "q3", "gender"]

# ── BUILD METADATA ────────────────────────────────────────────────────────
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)

for col in real.columns:
    if col in ORDINAL_ORDER:
        metadata.update_column(
            column_name=col,
            sdtype="categorical",
            order=ORDINAL_ORDER[col],
        )
    else:
        metadata.update_column(column_name=col, sdtype="categorical")

print("Metadata with ordinal ordering built.")

# ── FIT GAUSSIAN COPULA ───────────────────────────────────────────────────
synthesizer = GaussianCopulaSynthesizer(
    metadata,
    default_distribution="gaussian_kde",
)
synthesizer.fit(real)
print("GaussianCopula fitted on real data.")

# ── GENERATE ──────────────────────────────────────────────────────────────
synth = synthesizer.sample(num_rows=N_SYNTH)
print(f"Generated: {synth.shape}")

# ── SAVE ──────────────────────────────────────────────────────────────────
synth.to_excel("exp3_copula_global.xlsx", index=False)
print("Saved: exp3_copula_global.xlsx")

print("\nIB4 distribution:")
print(synth["IB4"].value_counts().sort_index())
