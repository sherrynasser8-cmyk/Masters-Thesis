"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 2 — CTGAN (Global)                                           ║
║  Dataset: Sustainable Apparel Consumer Behaviour Survey (34 columns)    ║
║  Library: SDV (Synthetic Data Vault)                                     ║
║  Method: Conditional GAN · no segmentation · 300 epochs                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Install: pip install sdv openpyxl

All 34 columns are already integer-coded (demographics 1-N category codes,
Likert items 1-5).
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

REAL_FILE = "REAL_DATA_APPAREL.xlsx"
N_SYNTH   = 384

real = pd.read_excel(REAL_FILE)
print(f"Real data: {real.shape}")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)
for col in real.columns:
    metadata.update_column(column_name=col, sdtype="categorical")
print("Metadata built — all 34 columns set to categorical.")

synthesizer = CTGANSynthesizer(metadata, epochs=300, batch_size=500, verbose=True)
synthesizer.fit(real)
print("CTGAN fitted on real data.")

synth = synthesizer.sample(num_rows=N_SYNTH)
print(f"Generated: {synth.shape}")

synth.to_excel("exp2_ctgan_global.xlsx", index=False)
print("Saved: exp2_ctgan_global.xlsx")

print("\nPI2 (target) distribution:")
print(synth["PI2"].value_counts().sort_index())
