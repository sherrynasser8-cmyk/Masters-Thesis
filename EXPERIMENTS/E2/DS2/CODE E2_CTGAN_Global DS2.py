"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 2 — CTGAN (Global)                                           ║
║  Dataset: Vietnam Online Privacy Survey                                  ║
║  Library: SDV (Synthetic Data Vault)                                     ║
║  Method: Conditional GAN · no segmentation · 300 epochs                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Install: pip install sdv openpyxl
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ── LOAD REAL DATA ────────────────────────────────────────────────────────
REAL_FILE  = "REAL_DATA_PRIVACY.xlsx"
N_SYNTH    = 467

real = pd.read_excel(REAL_FILE)
print(f"Real data: {real.shape}")

# ── BUILD METADATA ────────────────────────────────────────────────────────
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)

for col in real.columns:
    metadata.update_column(column_name=col, sdtype="categorical")

print("Metadata built — all columns set to categorical.")

# ── FIT CTGAN ─────────────────────────────────────────────────────────────
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    verbose=True,
)
synthesizer.fit(real)
print("CTGAN fitted on real data.")

# ── GENERATE ──────────────────────────────────────────────────────────────
synth = synthesizer.sample(num_rows=N_SYNTH)
print(f"Generated: {synth.shape}")

# ── SAVE ──────────────────────────────────────────────────────────────────
synth.to_excel("exp2_ctgan_global.xlsx", index=False)
print("Saved: exp2_ctgan_global.xlsx")

print("\nIB4 distribution:")
print(synth["IB4"].value_counts().sort_index())
