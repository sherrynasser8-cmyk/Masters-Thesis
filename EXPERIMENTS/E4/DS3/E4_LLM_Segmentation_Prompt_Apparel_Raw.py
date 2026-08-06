"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 4 — LLM + SEGMENTATION (Construct-Anchored Persona Prompt)   ║
║  Dataset: Sustainable Apparel Consumer Behaviour Survey (34 columns)    ║
║  Model: Claude (Anthropic)                                               ║
║  Method: K-Means (K=5) personas + two-step construct-anchored reasoning  ║
╚══════════════════════════════════════════════════════════════════════════╝

Combines the SEGMENTATION conditioning from the original E4 with the
CONSTRUCT-ANCHORED REASONING fix from the improved E1 (see
E1_LLM_Baseline_Prompt.py for the rationale).

5 K-Means segments (computed from real data):
  Segment 0 (n≈78,  20%): "Disengaged Consumers"        — all constructs ≈2.1-2.5
  Segment 1 (n≈94,  24%): "Engaged Mainstream Adopters"  — all constructs ≈4.0-4.3
  Segment 2 (n≈98,  26%): "Moderately Aware Consumers"   — all constructs ≈2.8-3.5
  Segment 3 (n≈38,  10%): "Eco-Pragmatic Buyers"         — EA/PI/BB high (≈4.4-4.7),
                                                            but MOT low (≈2.9)
  Segment 4 (n≈76,  20%): "Committed Sustainability Champions" — all constructs ≈4.1-4.9
"""

import json, os, random, time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from anthropic import Anthropic

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

client = Anthropic()
MODEL  = "claude-sonnet-4-6"

# ── LOAD REAL DATA (34 columns, already integer-coded) ─────────────────────
REAL_FILE = "REAL_DATA_APPAREL.xlsx"
real = pd.read_excel(REAL_FILE)
COLS = list(real.columns)

CONSTRUCTS = {
    "PK": ["PK1","PK2","PK3","PK4"], "CA": ["CA1","CA2","CA3","CA4"],
    "EA": ["EA1","EA2","EA3","EA4"], "PV": ["PV1","PV2","PV3","PV4"],
    "MOT": ["MOT1","MOT2","MOT3","MOT4"], "PI": ["PI1","PI2","PI3","PI4"],
    "BB": ["BB1","BB2","BB3","BB4"],
}
DEMO = ["gender","age","education","occupation","city","income"]

# ── K-MEANS SEGMENTATION (K=5) ───────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=SEED, n_init=50)
real["cluster"] = km.fit_predict(real[COLS].values)

shares = {c: (real["cluster"]==c).sum()/len(real) for c in range(5)}
n_per  = {}; rem = 384
for c in range(4):
    n_per[c] = int(round(shares[c]*384)); rem -= n_per[c]
n_per[4] = rem
print("Segment shares:", {k:f"{v:.2f}" for k,v in shares.items()})

SEGMENT_NAMES = [
    "Disengaged Consumers",
    "Engaged Mainstream Adopters",
    "Moderately Aware Consumers",
    "Eco-Pragmatic Buyers",
    "Committed Sustainability Champions",
]

DEMO_MAPS = {
    "gender": {1:"Male", 2:"Female"},
    "age": {1:"18-24", 2:"25-31", 3:"32-38", 4:"38+"},
    "education": {1:"Undergraduate", 2:"Graduate", 3:"Post-graduate", 4:"Professional"},
    "occupation": {1:"Student", 2:"Employee", 3:"Businessman", 4:"Homemaker"},
    "city": {1:"Lucknow", 2:"Kanpur", 3:"Meerut", 4:"Varanasi", 5:"Agra"},
    "income": {1:"<50,000 INR", 2:"50,000-100,000 INR", 3:">100,000 INR"},
}

# ── BUILD PER-SEGMENT CONSTRUCT-LEVEL + DEMOGRAPHIC PROFILES ───────────────
def build_persona(cluster_id):
    sub = real[real["cluster"] == cluster_id]
    p = {"segment_name": SEGMENT_NAMES[cluster_id]}
    for cname, items in CONSTRUCTS.items():
        p[cname] = round(sub[items].values.mean())
    for d in DEMO:
        mode_val = int(sub[d].mode().iloc[0])
        p[d] = DEMO_MAPS[d][mode_val]
    return p

personas = {c: build_persona(c) for c in range(5)}
for c, p in personas.items():
    print(f"Segment {c} ({p['segment_name']}): "
          f"PK={p['PK']} CA={p['CA']} EA={p['EA']} PV={p['PV']} MOT={p['MOT']} "
          f"PI={p['PI']} BB={p['BB']}")

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────────────
PROMPT_E4 = """You are simulating ONE respondent to a survey on consumer behaviour and
consumption patterns of SUSTAINABLE APPAREL.

This respondent belongs to the following CONSUMER SEGMENT, derived from
real survey data via K-Means clustering:

  Segment name: {segment_name}
  Gender:       {gender}
  Age:          {age}
  Education:    {education}
  Occupation:   {occupation}
  City:         {city}
  Income:       {income}

This segment's TYPICAL construct-level attitudes (1-5 scale, empirically
derived from real respondents in this segment) are:
  Product Knowledge (PK):        {PK}
  Consumer Awareness (CA):       {CA}
  Environmental Awareness (EA):  {EA}
  Perceived Value (PV):          {PV}
  Motivation (MOT):               {MOT}
  Purchase Intention (PI):        {PI}
  Buying Behaviour (BB):          {BB}

==================================================================
STEP-BY-STEP REASONING (do this internally BEFORE writing the JSON answer)
==================================================================

STEP 1 — Treat the seven construct-level values above as this respondent's
ANCHOR values. Do not deviate from the segment profile by more than ±1.

STEP 2 — For EACH item belonging to a construct, output a value EQUAL to
that construct's anchor in MOST cases, with occasional ±1 natural variation
(real data shows within-construct correlations r≈0.60-0.86 — do NOT make
items within a construct independent of each other). Exception: MOT2 (social
media influence) may deviate more from the MOT anchor than other MOT items.

STEP 3 — Demographics MUST match the segment profile: gender={gender},
age={age}, education={education}, occupation={occupation}, city={city},
income={income}.

==================================================================
SURVEY ITEMS TO ANSWER (return integer values only, 1-5 for Likert items)
==================================================================

DEMOGRAPHICS (return as integers matching these codes):
gender: 1=Male, 2=Female
age: 1="18-24", 2="25-31", 3="32-38", 4="38+"
education: 1=Undergraduate, 2=Graduate, 3=Post-graduate, 4=Professional
occupation: 1=Student, 2=Employee, 3=Businessman, 4=Homemaker
city: 1=Lucknow, 2=Kanpur, 3=Meerut, 4=Varanasi, 5=Agra
income: 1="<50,000 INR", 2="50,000-100,000 INR", 3=">100,000 INR"

PRODUCT KNOWLEDGE (PK) — anchor {PK}:
PK1: Clothing made from organic fibers is stylish.
PK2: Clothing made from organic fibers has attractive designs.
PK3: Clothing made from organic fibers is durable.
PK4: Clothing made from organic fibers has good quality.

CONSUMER AWARENESS (CA) — anchor {CA}:
CA1: I am aware that garments made from recycled materials will biodegrade instead of adding to a landfill.
CA2: I am aware that companies can follow environmentalism/social policy standards in clothing production.
CA3: I am aware that second-hand or upcycled clothes can be sustainable choices.
CA4: I am aware that products made without animal tissue are considered more sustainable.

ENVIRONMENTAL AWARENESS (EA) — anchor {EA}:
EA1: I am aware that dyeing/finishing processes for textiles can result in significant water waste.
EA2: I am aware that manufacturing synthetic fibers like polyester can cause environmental pollution.
EA3: I am aware that air pollution can occur during common textile dyeing processes.
EA4: I am aware that cultivating raw materials for garments (e.g. cotton) can have significant environmental impact.

PERCEIVED VALUE (PV) — anchor {PV}:
PV1: I believe that sustainable clothing is well-made and worth the money.
PV2: I feel good when I purchase sustainable apparel.
PV3: Sustainable clothing offers a unique style to me.
PV4: I believe that sustainable clothing helps in saving resources.

MOTIVATION (MOT) — anchor {MOT} (MOT2 may deviate more):
MOT1: I feel good about contributing to the environment through sustainable clothing.
MOT2: My friends' opinions on Instagram influence my apparel purchase decisions.
MOT3: I feel guilty that animals have died because of human consumption habits.
MOT4: I believe that clothing made from organic fibers is beneficial for my skin.

PURCHASE INTENTION (PI) — anchor {PI}:
PI1: I am willing to pay more for clothing from environmentally/socially responsible brands.
PI2: In the future, I intend to purchase environmentally sustainable apparel.
PI3: I would consider buying environmentally friendly apparel because it is less polluting.
PI4: I would consider switching to environmentally friendly apparel brands for ecological reasons.

BUYING BEHAVIOUR (BB) — anchor {BB}:
BB1: When choosing between two similar items, I prefer the one made from less environmentally harmful materials.
BB2: I prioritize apparel that can be worn long-term over trendy apparel that quickly goes out of style.
BB3: I prefer to buy clothing made of organically grown natural fibers.
BB4: I prefer apparel with environmentally friendly labeling or packaging.

==================================================================
OUTPUT
==================================================================
Return ONLY a JSON object with these 34 keys and integer values.
No explanation, no markdown.

{{"gender":...,"age":...,"education":...,"occupation":...,"city":...,"income":...,
 "PK1":...,"PK2":...,"PK3":...,"PK4":...,
 "CA1":...,"CA2":...,"CA3":...,"CA4":...,
 "EA1":...,"EA2":...,"EA3":...,"EA4":...,
 "PV1":...,"PV2":...,"PV3":...,"PV4":...,
 "MOT1":...,"MOT2":...,"MOT3":...,"MOT4":...,
 "PI1":...,"PI2":...,"PI3":...,"PI4":...,
 "BB1":...,"BB2":...,"BB3":...,"BB4":...}}
"""

DEMO_REVERSE = {d: {v:k for k,v in m.items()} for d,m in DEMO_MAPS.items()}


def call_llm_e4(cluster_id):
    persona = personas[cluster_id]
    prompt = PROMPT_E4.format(**persona)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    row = json.loads(text.strip())
    # Demographics may come back as labels; convert to codes if needed
    for d in DEMO:
        v = row.get(d)
        if isinstance(v, str) and v in DEMO_REVERSE[d]:
            row[d] = DEMO_REVERSE[d][v]
    return row


if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY_HERE"

    weights = [shares[c] for c in range(5)]
    rows, failures = [], 0
    N = 384

    for i in range(N):
        cluster_id = random.choices(range(5), weights=weights, k=1)[0]
        for attempt in range(3):
            try:
                row = call_llm_e4(cluster_id)
                rows.append({c: row.get(c, 3) for c in COLS})
                break
            except Exception as e:
                if attempt == 2:
                    failures += 1
                    print(f"Row {i} failed: {e}")
                else:
                    time.sleep(1)

        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{N} | failures: {failures}")

    df = pd.DataFrame(rows)[COLS]
    df.to_excel("exp4_llm_segmentation.xlsx", index=False)
    print(f"\nSaved exp4_llm_segmentation.xlsx  ({len(df)} rows, {failures} failures)")
