# Evaluating Synthetic Consumer Survey Data for Scalable Marketing Research

Master's thesis and associated book chapter research repository — German University in Cairo (GUC), Faculty of Management Technology.

**Author:** Sherry Nasser
**Co-author (book chapter):** Noha Amer

## Overview

This project investigates whether AI-generated and statistically-generated synthetic survey respondents can substitute for real human respondents in scalable marketing research. The work follows Peffers' Design Science Research Methodology (DSRM) and is theoretically grounded in Data Quality Theory (Wang & Strong, 1996) and Consumer Heterogeneity Theory (Wedel & Kamakura, 2000).

Two related outputs come out of this research:

- **Book chapter:** *"Evaluating the Validity of Generative AI as Synthetic Response for Scalable Marketing Research: An Experimental Analysis"* — submitted to the Springer edited volume *Positive Performativity and Transformative Management (Volume II)*, GUC Faculty of Management Technology Conference (May 2026). Currently under Major Revision.
- **Master's thesis:** A broader six-experiment comparison (LLM, CTGAN, and Gaussian copula methods, each in zero-shot and segmented variants) evaluated for fidelity, utility, and privacy, plus AHP-based expert ranking and fsQCA validity analysis.

## Datasets

| ID | Domain | Source | n |
|----|--------|--------|---|
| DS1 | Luxury products | Rahman (2025), Mendeley Data | 272 |
| DS2 | Consumer privacy attitudes (Vietnam) | Khoa, Anh & Huynh (2024), Data in Brief | 467 |
| DS3 | Sustainable apparel (India) | Mishra et al. (2023), Data in Brief | 384 |

## Repository Structure

- **`REAL DATASETS/`** — Original real-respondent datasets (DS1, DS2, DS3) used as ground truth.
- **`EXPERIMENTS/`** — Synthetic data generation experiments, including zero-shot role-play prompting (E1)implemented via Claude Sonnet, CTGAN (E2), Gaussian Copula (E3), Segment-based role prompting (E4)implemented via Claude Sonnet,Segment-based Gaussian Copula (E5) and Segment-based CTGAN (E6) using K-Means-derived personas (K=5, seed=42). 
- **`AHP/`** — Analytic Hierarchy Process expert survey materials and results (5-expert canonical run, CR=0.042), used for the decision-ranking objective across the six thesis experiments.
- **`Evaluation/`** — Fidelity (Pearson L2 distance), utility (TSTR Accuracy), and privacy (DCR) evaluation code and results.
- **`Correlation test/`** — Pearson correlation analysis across all 18 experiment–dataset combinations examining the fidelity–utility-privacy tradeoff.
- **`Prototype/`** — Interactive HTML prototype (`SynthGen_Prototype.html`) implementing Saaty matrix math with live consistency ratio (CR) computation and a six-experiment leaderboard.

## Key Methods

- **Fidelity:** Pearson L2 distance (minimize)
- **Utility:** Train-Synthetic-Test-Real (TSTR) Accuracy  (maximize)
- **Privacy:** Distance to Closest Record — DCR (maximize)
- **AHP:** Saaty pairwise comparison matrices with consistency ratio validation
- **fsQCA:** Configurational comparison of real vs. synthetic conditions (Pappas & Woodside, 2021)

Seed = 42 applied consistently across all experiments for reproducibility.

## Key Findings

- Segmentation (segment/cluster-based generation) consistently improves fidelity and utility but reduces privacy (DCR) — a structural tradeoff confirmed across datasets (r = 0.529, p = 0.024).
- The magnitude of the fidelity–privacy tradeoff is dataset-dependent: luxury data shows the steepest privacy cost, privacy-attitude data shows the largest fidelity gain, and apparel data shows the most modest movement.
- fsQCA reveals that consistency metrics alone can mask structural divergence between real and synthetic configurations — configurational (Jaccard) comparison is also needed.

## Limitations

- Single-run-per-experiment design (seed=42 only); no confidence intervals or significance testing yet. Multi-seed replication is a priority for future work.
- K=5 cluster count was fixed a priori without formal validation; this is planned as a future methodological improvement.

## Citations

Key theoretical and methodological references: Wang & Strong (1996), Wedel & Kamakura (2000), Esteban, Hyland & Rätsch (2017), Mendelevitch & Lesh (2021), Platzer & Reutterer (2021), Pappas & Woodside (2021).
