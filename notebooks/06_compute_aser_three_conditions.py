"""
Compute full ASER (4-entity, all 1952 rows) for all three conditions,
using the same population and fallback convention as the WER comparison.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

XCAS = Path("/work/arun/shreyasv/xcas-ga")

POS_KEYWORDS = ["final","base","downwind","crosswind","departing","departure",
                "entering","inbound","rolling","taxi"]
RWY_RE = re.compile(
    r"(?:runway\s+)?(?<![\w])(two six|2\s*6|26|eight|0\s*8|08|zero eight)(?![\w])",
    re.I
)
AIRPORT_RE = re.compile(r"\bbutler\b", re.I)

def norm_rwy(s):
    if not s: return None
    s = s.lower().replace(" ", "")
    if s in ("twosix","26"): return "26"
    if s in ("eight","08","zeroeight"): return "08"
    return None

def extract_position(text):
    t = str(text).lower()
    return next((kw for kw in POS_KEYWORDS if kw in t), None)

def extract_runway(text):
    t = str(text).lower()
    pos_match = None
    for kw in POS_KEYWORDS:
        idx = t.rfind(kw)
        if idx > (pos_match[1] if pos_match else -1):
            pos_match = (kw, idx)
    search_start = pos_match[1] if pos_match else 0
    m = RWY_RE.search(t[search_start:])
    return norm_rwy(m.group(1)) if m else None

def extract_entities(text):
    t = str(text).lower()
    callsign_tokens = re.findall(
        r"\b(november|alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|"
        r"juliet|kilo|lima|mike|oscar|papa|quebec|romeo|sierra|tango|uniform|"
        r"victor|whiskey|xray|yankee|zulu|\d)\b", t)
    return {
        "runway": extract_runway(text),
        "position": extract_position(text),
        "airport": bool(AIRPORT_RE.search(t)),
        "callsign": " ".join(callsign_tokens) if callsign_tokens else None,
    }

_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2",
                                           cache_folder=str(XCAS/"cache/hf"))
    return _embed_model

def semwer(h, r):
    h, r = str(h), str(r)
    if not h.strip() or not r.strip():
        return 1.0
    m = get_embed_model()
    e = m.encode([h, r], convert_to_numpy=True, show_progress_bar=False)
    from sklearn.metrics.pairwise import cosine_similarity
    return float(1 - cosine_similarity(e[0:1], e[1:2])[0][0])

WEIGHTS_FULL = {"runway": 3.0, "position": 1.5, "airport": 1.0, "callsign": 0.5}

def eer(h_ent, r_ent, weights=WEIGHTS_FULL):
    tw, te = 0.0, 0.0
    for k, w in weights.items():
        rv = r_ent.get(k)
        if rv is None or rv is False:
            continue
        hv = h_ent.get(k)
        match = (hv == rv) if k != "callsign" else (
            hv is not None and rv is not None and
            len(set(hv.split()) & set(rv.split())) >= max(1, len(rv.split())//2)
        )
        tw += w
        te += 0 if match else w
    return te / tw if tw > 0 else 0.0

def full_aser(h, r, alpha=0.4):
    h_ent, r_ent = extract_entities(h), extract_entities(r)
    return alpha*semwer(h, r) + (1-alpha)*eer(h_ent, r_ent)

print("Loading data...")
df_main = pd.read_csv(XCAS/"outputs"/"pcas_pipeline_results_v3.csv")
df_abl  = pd.read_csv(XCAS/"outputs"/"pcas_ablation_no_context.csv")
assert len(df_main) == len(df_abl)

def get_pcas_text(row):
    if row["path_v2"] == "fast_bypass":
        return row["asr_text"]
    if str(row["verify_status_v2"]).startswith("PASS"):
        return row["llm_output"]
    return row["asr_text"]
df_main["pcas_text"] = df_main.apply(get_pcas_text, axis=1)

print("Computing full ASER — ASR only (all 1952 rows)...")
aser_asr = df_main.apply(lambda r: full_aser(r["asr_text"], r["synthetic_reference"]), axis=1)

print("Computing full ASER — LLM, no ADS-B context (all 1952 rows)...")
aser_noctx = df_abl.apply(lambda r: full_aser(r["llm_output_noctx"], r["synthetic_reference"]), axis=1)

print("Computing full ASER — PCAS, full pipeline (all 1952 rows)...")
aser_pcas = df_main.apply(lambda r: full_aser(r["pcas_text"], r["synthetic_reference"]), axis=1)

print(f"\n{'='*60}")
print(f"FULL ASER (4-entity) — full October corpus, n={len(df_main)}, all rows")
print(f"{'='*60}")
print(f"  ASR only (baseline)        : {aser_asr.mean():.3f}")
print(f"  + LLM, no ADS-B context    : {aser_noctx.mean():.3f}")
print(f"  + LLM + ADS-B (PCAS)       : {aser_pcas.mean():.3f}")