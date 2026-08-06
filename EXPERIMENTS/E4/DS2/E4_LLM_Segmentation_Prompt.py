"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 4 — LLM + SEGMENTATION PROMPT                               ║
║  Dataset: Vietnam Online Privacy Survey                                  ║
║  Model: Claude (Anthropic)                                               ║
║  Method: K-Means (K=5) → persona-conditioned prompt per segment          ║
╚══════════════════════════════════════════════════════════════════════════╝

Steps:
  1. Cluster real data into 5 segments using K-Means
  2. Build a natural-language privacy persona for each segment
  3. For each synthetic respondent, pick a segment (proportional to share)
  4. Inject the segment profile into the prompt and call Claude
"""

import json, os, random, time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from anthropic import Anthropic

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

client = Anthropic()
MODEL  = "claude-sonnet-4-6"

# ── LOAD & ENCODE REAL DATA ───────────────────────────────────────────────
REAL_FILE = "REAL_DATA_PRIVACY.xlsx"
real = pd.read_excel(REAL_FILE)
COLS = list(real.columns)

real_enc = pd.DataFrame()
for col in COLS:
    le = LabelEncoder()
    le.fit(real[col].astype(str))
    real_enc[col] = le.transform(real[col].astype(str))

# ── K-MEANS SEGMENTATION (K=5) ────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=SEED, n_init=50)
real["cluster"] = km.fit_predict(real_enc.values)

shares = {c: (real["cluster"]==c).sum()/len(real) for c in range(5)}
n_per  = {}; rem = 467
for c in range(4):
    n_per[c] = int(round(shares[c]*467)); rem -= n_per[c]
n_per[4] = rem
print("Segment shares:", {k:f"{v:.2f}" for k,v in shares.items()})

# ── BUILD SEGMENT PROFILES ────────────────────────────────────────────────
feature_cols = [c for c in COLS]   # all 31 columns

SEGMENT_NAMES = [
    "Privacy-Cautious Infrequent Shopper",
    "Moderate Privacy Concern Regular Buyer",
    "High Privacy Concern Frequent Shopper",
    "Privacy-Relaxed Occasional Browser",
    "Privacy-Aware Engaged Consumer",
]

def build_profile(cluster_id):
    sub = real[real["cluster"] == cluster_id]
    return {col: sub[col].mode().iloc[0] for col in feature_cols}

profiles = {c: build_profile(c) for c in range(5)}

# ── PERSONA DESCRIPTION BUILDER ───────────────────────────────────────────
Q4_MAP = {2:"1-2 times/month", 3:"3-5 times/month",
           4:"6-10 times/month", 5:"More than 10 times/month"}
Q5_MAP = {1:"Identity theft", 2:"Data breaches",
          3:"Tracking of browsing history", 4:"Sharing data with third parties",
          5:"Lack of transparency about data practices"}
GENDER_MAP  = {1:"Male", 2:"Female"}
AGE_MAP     = {1:"18-25", 2:"26-35", 3:"36-45", 4:">45"}
OCC_MAP     = {1:"Housewife", 2:"Student", 3:"Office worker",
               4:"Lecturer", 5:"Government employee"}

def describe_persona(cluster_id):
    p = profiles[cluster_id]
    likert_avg = round(
        np.mean([p.get(c, 3) for c in
                 ["COL1","COL2","COL3","COL4",
                  "ERR1","ERR2","ERR3","ERR4",
                  "USU1","USU2","USU3","USU4",
                  "IMA1","IMA2","IMA3"]]), 1
    )
    trust_avg = round(np.mean([p.get(c,3) for c in ["OT1","OT2","OT3","OT4"]]), 1)
    intent_avg= round(np.mean([p.get(c,3) for c in ["IB1","IB2","IB3","IB4"]]), 1)
    return {
        "segment_name":     SEGMENT_NAMES[cluster_id],
        "gender":           GENDER_MAP.get(int(p.get("gender",2)), "Female"),
        "age_group":        AGE_MAP.get(int(p.get("age",2)), "26-35"),
        "occupation":       OCC_MAP.get(int(p.get("occupation",3)), "Office worker"),
        "shopping_freq":    Q4_MAP.get(int(p.get("q4",3)), "3-5 times/month"),
        "main_concern":     Q5_MAP.get(int(p.get("q5",2)), "Data breaches"),
        "privacy_concern":  f"{likert_avg}/5 average privacy concern",
        "online_trust":     f"{trust_avg}/5 average online trust",
        "purchase_intent":  f"{intent_avg}/5 average purchase intention",
    }

personas = {c: describe_persona(c) for c in range(5)}

# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────
PROMPT_E4 = """You are a single respondent to a Vietnamese online privacy and consumer behaviour survey.

You match this CONSUMER SEGMENT profile:
  Segment name:         {segment_name}
  Gender:               {gender}
  Age group:            {age_group}
  Occupation:           {occupation}
  Shopping frequency:   {shopping_freq}
  Main privacy concern: {main_concern}
  Privacy concern level:{privacy_concern}
  Online trust level:   {online_trust}
  Purchase intention:   {purchase_intent}

CONTEXT: This study examines how companies' privacy practices influence online trust and purchase intentions in Vietnam.

RULES:
1. q1, q2, q3 must ALWAYS be 1 (Yes).
2. q4 must be 2, 3, 4, or 5 — matching shopping frequency: {shopping_freq}.
3. q5 must be 1, 2, 3, 4, or 5 — reflecting: {main_concern}.
4. For all Likert items: use 1, 2, 3, 4, or 5. Scores should reflect privacy concern level: {privacy_concern}.
5. OT1-OT4 (trust) should reflect: {online_trust}.
6. IB1-IB4 (purchase intention) should reflect: {purchase_intent}.
7. Demographics MUST match: gender={gender}, age={age_group}, occupation={occupation}.
8. Allow natural variation — not every item needs to be identical to the average.
9. Return ONLY a JSON object. No explanation, no markdown.

SURVEY QUESTIONS (return integer values only):

Screening: q1=1, q2=1, q3=1 (fixed)
q4: Shopping frequency (2=1-2x/month, 3=3-5x, 4=6-10x, 5=>10x)
q5: Main concern (1=Identity theft, 2=Data breaches, 3=Tracking, 4=Sharing data, 5=Lack transparency)

Likert 1-5 (1=Strongly disagree, 5=Strongly agree):
COL1-COL4: Collection concerns
ERR1-ERR4: Error/accuracy concerns
USU1-USU4: Unauthorized secondary use concerns
IMA1-IMA3: Improper access concerns
OT1-OT4:   Online trust
IB1-IB4:   Purchase intention

Demographics:
gender (1=Male, 2=Female), age (1=18-25,2=26-35,3=36-45,4=>45), occupation (1-5)

Return ONLY valid JSON with all 31 keys:
{{"q1":1,"q2":1,"q3":1,"q4":...,"q5":...,"COL1":...,...,"IB4":...,"gender":...,"age":...,"occupation":...}}
"""

# ── GENERATION FUNCTION ───────────────────────────────────────────────────
def call_llm_e4(cluster_id):
    persona = personas[cluster_id]
    prompt  = PROMPT_E4.format(**persona)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    return json.loads(text.strip())

# ── GENERATE 467 RESPONDENTS ──────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY_HERE"

    weights = [shares[c] for c in range(5)]
    rows, failures = [], 0
    N = 467

    for i in range(N):
        cluster_id = random.choices(range(5), weights=weights, k=1)[0]
        for attempt in range(3):
            try:
                row = call_llm_e4(cluster_id)
                row["q1"] = row["q2"] = row["q3"] = 1
                rows.append({c: row.get(c, 1) for c in feature_cols})
                break
            except Exception as e:
                if attempt == 2:
                    failures += 1
                    print(f"Row {i} failed: {e}")
                else:
                    time.sleep(1)

        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{N} | failures: {failures}")

    df = pd.DataFrame(rows)[feature_cols]
    df.to_excel("exp4_llm_segmentation.xlsx", index=False)
    print(f"\nSaved exp4_llm_segmentation.xlsx  ({len(df)} rows, {failures} failures)")
