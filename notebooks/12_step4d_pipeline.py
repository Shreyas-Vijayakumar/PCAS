"""
Step 4d — Fast-Bypass / LLM Correction / Verification pipeline.
Input : outputs/utterance_synthetic_pairs.csv (1952 joined pairs)
Output: outputs/pcas_pipeline_results.csv
Model : Mistral-7B-Instruct-v0.3 (same as Approach 1)
"""

import re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

XCAS      = Path("/work/arun/shreyasv/xcas-ga")
IN_CSV    = XCAS / "outputs" / "utterance_synthetic_pairs.csv"
OUT_CSV   = XCAS / "outputs" / "pcas_pipeline_results.csv"

import os
os.environ["HF_HOME"] = str(XCAS / "cache/hf")

TAU_BYPASS  = 0.15
TAU_VERIFY  = 0.30
LEN_EXPLOSION_RATIO = 2.5

# ── ASER components (entity extraction + weighted EER + SemWER) ──────────────
RWY_RE = re.compile(
    r"(?:runway\s+)?(?<![\w])(two six|2\s*6|26|eight|0\s*8|08|zero eight)(?![\w])",
    re.I
)
POS_KEYWORDS = ["final","base","downwind","crosswind","departing","departure",
                "entering","inbound","rolling","taxi"]
AIRPORT_RE = re.compile(r"\bbutler\b", re.I)

def norm_rwy(s):
    if not s: return None
    s = s.lower().replace(" ", "")
    if s in ("twosix","26"): return "26"
    if s in ("eight","08","zeroeight"): return "08"
    return None

def extract_entities(text):
    t = text.lower()
    rwy_m = RWY_RE.search(t)
    pos   = next((kw for kw in POS_KEYWORDS if kw in t), None)
    airport = bool(AIRPORT_RE.search(t))
    callsign_tokens = re.findall(
        r"\b(november|alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|"
        r"juliet|kilo|lima|mike|oscar|papa|quebec|romeo|sierra|tango|uniform|"
        r"victor|whiskey|xray|yankee|zulu|\d)\b", t)
    return {
        "runway": norm_rwy(rwy_m.group(1)) if rwy_m else None,
        "position": pos,
        "airport": airport,
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
    if not h.strip() or not r.strip():
        return 1.0
    m = get_embed_model()
    e = m.encode([h, r], convert_to_numpy=True, show_progress_bar=False)
    from sklearn.metrics.pairwise import cosine_similarity
    return float(1 - cosine_similarity(e[0:1], e[1:2])[0][0])

def eer(h_ent, r_ent, weights):
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

def aser(h, r, weights, alpha=0.4):
    h_ent, r_ent = extract_entities(h), extract_entities(r)
    e = eer(h_ent, r_ent, weights)
    s = semwer(h, r)
    return alpha*s + (1-alpha)*e

WEIGHTS_FASTBYPASS = {"runway": 3.0, "position": 1.5}
WEIGHTS_FULL       = {"runway": 3.0, "position": 1.5, "airport": 1.0, "callsign": 0.5}

# ── Confidence conversion ──────────────────────────────────────────────────────
def conf_to_prob(mean_log_prob):
    if pd.isna(mean_log_prob):
        return 0.0
    return float(np.exp(mean_log_prob))

# ── Load Mistral (same setup as Approach 1) ───────────────────────────────────
print("Loading Mistral-7B-Instruct-v0.3...")
from transformers import AutoTokenizer, AutoModelForCausalLM
LLM_ID = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(LLM_ID)
tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    LLM_ID, torch_dtype=torch.float16, device_map="auto",
    attn_implementation="eager",
).eval()
print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

SYSTEM_PROMPT = """You are an aviation radio communication corrector for Butler County Airport (KBTP), a non-towered general aviation airport.

Correct the ASR transcript to standard CTAF phraseology.
Rules:
1. Return ONLY the corrected transcript, nothing else
2. Fix mishearings of "Butler" airport name
3. The active runway is {runway} — any runway mentioned must match this
4. Preserve words you are not confident are wrong
5. Never invent information absent from the input"""

def correct_with_llm(asr_text, runway, tail, phase, range_nm, synthetic_ref):
    sys_msg = SYSTEM_PROMPT.format(runway=runway)
    user_msg = (f"ADS-B context: Aircraft {tail}, Phase {phase}, "
                f"Range {range_nm:.1f}NM, Runway {runway}\n"
                f"Synthetic reference: \"{synthetic_ref}\"\n\n"
                f"Raw ASR transcript: \"{asr_text}\"\n\nCorrected transcript:")
    messages = [{"role":"system","content":sys_msg},
                {"role":"user","content":user_msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=80, temperature=0.1,
                           do_sample=True, repetition_penalty=1.3,
                           pad_token_id=tok.eos_token_id)
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(new_tok, skip_special_tokens=True).strip()
    return raw.split("\n")[0].strip().strip('"').strip()

# ── Verification ────────────────────────────────────────────────────────────────
def verify(corrected, asr_text, runway, tail, synthetic_ref):
    rwy_m = RWY_RE.search(corrected.lower())
    if rwy_m and norm_rwy(rwy_m.group(1)) != runway:
        return False, "runway_mismatch"
    tail_tokens = set(str(tail).upper())
    corr_upper = corrected.upper()
    callsign_hit = any(c in corr_upper for c in tail_tokens if c.isdigit())
    if not callsign_hit and len(corrected.split()) > 4:
        pass  # soft check only — many valid corrections omit full tail
    score = aser(corrected, synthetic_ref, WEIGHTS_FULL)
    if score > TAU_VERIFY:
        return False, f"aser_high:{score:.3f}"
    if len(corrected.split()) > len(asr_text.split()) * LEN_EXPLOSION_RATIO:
        return False, "length_explosion"
    return True, f"PASS:aser={score:.3f}"

# ── Main loop ──────────────────────────────────────────────────────────────────
print("\nLoading utterance pairs...")
df = pd.read_csv(IN_CSV, dtype={"active_runway": str})
print(f"  {len(df)} pairs loaded")

results = []
n_bypass = n_llm = n_verify_pass = n_verify_fail = 0
t0 = time.time()

for i, row in df.iterrows():
    asr_text = row["asr_text"]
    runway   = row["active_runway"]
    synth    = row["synthetic_reference"]
    tail     = row["matched_tail"]
    phase    = row["matched_adsb_phase"]
    rng      = row["matched_range_nm"]
    conf_prob = conf_to_prob(row["confidence"])

    aser_fb = aser(asr_text, synth, WEIGHTS_FASTBYPASS)

    if conf_prob >= 0.95 and aser_fb <= TAU_BYPASS:
        n_bypass += 1
        results.append({**row.to_dict(), "path": "fast_bypass",
                        "aser_fastbypass": aser_fb, "final_output": asr_text,
                        "verify_status": "N/A (bypassed)"})
        continue

    n_llm += 1
    corrected = correct_with_llm(asr_text, runway, tail, phase, rng, synth)
    passed, reason = verify(corrected, asr_text, runway, tail, synth)

    if passed:
        n_verify_pass += 1
        final_output = corrected
    else:
        n_verify_fail += 1
        wav_path = XCAS / "data" / (
            f"10-22-20_audio/{row['wav_file']}" if row["day"] == "10-22-20"
            else f"corpus_audio/{row['day']}_audio/{row['wav_file']}"
        )
        final_output = str(wav_path)

    results.append({**row.to_dict(), "path": "llm_correction",
                    "aser_fastbypass": aser_fb, "llm_output": corrected,
                    "verify_status": reason, "final_output": final_output})

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(df)}]  bypass={n_bypass}  llm={n_llm}  "
              f"verify_pass={n_verify_pass}  verify_fail={n_verify_fail}  "
              f"  {elapsed:.0f}s elapsed")

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"STEP 4d COMPLETE  ({(time.time()-t0)/60:.1f} min)")
print(f"  Total utterances    : {len(df)}")
print(f"  Fast-bypass         : {n_bypass}  ({100*n_bypass/len(df):.1f}%)")
print(f"  Sent to LLM         : {n_llm}  ({100*n_llm/len(df):.1f}%)")
print(f"    Verify pass       : {n_verify_pass}")
print(f"    Verify fail (→audio fallback): {n_verify_fail}")
print(f"  Saved → {OUT_CSV}")