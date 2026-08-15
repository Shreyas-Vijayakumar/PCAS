"""
Step 4c — Join ASR utterances (keyword-filtered traffic pattern segments)
to their nearest ADS-B synthetic reference.

Input : outputs/corpus_traffic_pattern.jsonl   (2782 utterances, 16 days)
        outputs/synthetic_references_all_days.csv (570,929 classified pings)
        data/corpus_audio/{day}_audio/*.txt    (WAV start/end times, naive)
        data/10-22-20_audio/*.txt              (test set, same format)

Output: outputs/utterance_synthetic_pairs.csv
        one row per utterance with: asr_text, utterance_phase,
        matched ADS-B ping (Tail, range_nm, phase, active_runway,
        synthetic_reference), time_diff_s, match_quality flag
"""

import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XCAS         = Path("/work/arun/shreyasv/xcas-ga")
UTTER_JSONL  = XCAS / "outputs" / "corpus_traffic_pattern.jsonl"
SYNREF_CSV   = XCAS / "outputs" / "synthetic_references_all_days.csv"
OUT_CSV      = XCAS / "outputs" / "utterance_synthetic_pairs.csv"

CORPUS_AUDIO = XCAS / "data" / "corpus_audio"
TEST_AUDIO   = XCAS / "data" / "10-22-20_audio"

MATCH_TOLERANCE_S = 45     # nearest-ping tolerance, per Approach 1 precedent

# ── Utterance phase classification (your confirmed priority order) ───────────
PHASE_KEYWORDS = [
    (r"\bfinal\b",                                   "FINAL"),
    (r"\bbase\b",                                    "BASE"),
    (r"\bdownwind\b",                                "DOWNWIND"),
    (r"\bcrosswind\b",                                "CROSSWIND"),
    (r"\b(departing|departure|rolling)\b",            "DEPARTING"),
    (r"\b(entering|teardrop|straight-in|straight in)\b","PATTERN_ENTRY"),
]
INBOUND_DIST_RE = re.compile(
    r"\binbound\b.*?\b(ten|10|five|5|three|3)\b|"
    r"\b(ten|10|five|5|three|3)\b.*?\binbound\b", re.IGNORECASE
)
INBOUND_RE = re.compile(r"\binbound\b", re.IGNORECASE)
DIST_MAP = {"ten":"10NM","10":"10NM","five":"5NM","5":"5NM","three":"3NM","3":"3NM"}

def classify_utterance_phase(text: str) -> str | None:
    t = text.lower()
    for pat, label in PHASE_KEYWORDS:
        if re.search(pat, t):
            return label
    m = INBOUND_DIST_RE.search(t)
    if m:
        digit = next(g for g in m.groups() if g)
        return f"INBOUND_{DIST_MAP[digit]}"
    if INBOUND_RE.search(t):
        return "INBOUND_UNSPECIFIED"   # backfilled from ADS-B below
    return None

# ── Audio .txt sidecar parsing (matches notebook 01 exactly) ──────────────────
def parse_audio_txt(txt_path: Path) -> dict | None:
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        sm = re.search(r"Start Time:\s*(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[\.\d]*)", text)
        em = re.search(r"End Time:\s*(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[\.\d]*)", text)
        if not sm or not em:
            return None
        return {"start_time": pd.Timestamp(sm.group("dt")),
                "end_time":   pd.Timestamp(em.group("dt"))}
    except Exception:
        return None

def find_wav_dir(day_dir_name: str) -> Path:
    """day_dir_name already includes '_audio' suffix, e.g. '10-01-20_audio'."""
    candidate = CORPUS_AUDIO / day_dir_name
    if candidate.exists():
        return candidate
    if day_dir_name == "10-22-20_audio":
        return TEST_AUDIO
    return None

# ── Build per-WAV start_time lookup (cached) ──────────────────────────────────
_WAV_START_CACHE = {}
def get_wav_start(day_dir_name: str, wav_file: str):
    key = (day_dir_name, wav_file)
    if key in _WAV_START_CACHE:
        return _WAV_START_CACHE[key]
    wav_dir = find_wav_dir(day_dir_name)
    if wav_dir is None:
        _WAV_START_CACHE[key] = None
        return None
    txt_path = wav_dir / (Path(wav_file).stem + ".txt")
    if not txt_path.exists():
        _WAV_START_CACHE[key] = None
        return None
    info = parse_audio_txt(txt_path)
    _WAV_START_CACHE[key] = info["start_time"] if info else None
    return _WAV_START_CACHE[key]

# ── Load synthetic references (dtype guard applied) ───────────────────────────
print("Loading synthetic references...")
df_syn = pd.read_csv(SYNREF_CSV, dtype={"active_runway": str},
                     parse_dates=["timestamp"])
df_syn = df_syn.sort_values("timestamp").reset_index(drop=True)
print(f"  {len(df_syn):,} classified pings loaded")

def find_nearest_ping(day: str, abs_time: pd.Timestamp, utter_phase: str = None):
    """
    Nearest ADS-B ping, preferring aircraft whose classified phase
    matches the utterance's keyword-classified phase when multiple
    aircraft have pings within tolerance — this disambiguates between
    simultaneous aircraft rather than picking whichever is closest
    in time regardless of which aircraft is actually transmitting.
    """
    day_pings = df_syn[df_syn["day"] == day]
    if len(day_pings) == 0:
        return None

    window = day_pings[
        (day_pings["timestamp"] - abs_time).abs() <= pd.Timedelta(seconds=MATCH_TOLERANCE_S)
    ].copy()
    if len(window) == 0:
        return None

    window["time_diff_s"] = (window["timestamp"] - abs_time).abs().dt.total_seconds()

    # Prefer phase-matching candidates first
    if utter_phase and utter_phase != "INBOUND_UNSPECIFIED":
        phase_matches = window[window["phase"] == utter_phase]
        if len(phase_matches) > 0:
            idx = phase_matches["time_diff_s"].idxmin()
            return window.loc[idx]

    # Fallback: nearest in time among all candidates
    idx = window["time_diff_s"].idxmin()
    return window.loc[idx]

# ── Main join ──────────────────────────────────────────────────────────────────
print("\nJoining utterances to synthetic references...")
results = []
n_total = n_matched = n_backfilled = n_unmatched_phase = n_unmatched_ping = 0

with open(UTTER_JSONL) as f:
    for line in f:
        rec = json.loads(line)
        n_total += 1
        day_dir_name = rec["day"]                      # e.g. "10-01-20_audio"
        day_bare     = day_dir_name.replace("_audio", "")  # "10-01-20"
        wav_file     = rec["wav_file"]
        seg_start    = rec["start"]
        asr_text     = rec["text"]
        confidence   = rec.get("confidence")

        utter_phase = classify_utterance_phase(asr_text)
        if utter_phase is None:
            n_unmatched_phase += 1
            continue

        wav_start = get_wav_start(day_dir_name, wav_file)
        if wav_start is None:
            n_unmatched_ping += 1
            continue
        abs_time = wav_start + pd.Timedelta(seconds=seg_start)

        ping = find_nearest_ping(day_bare, abs_time, utter_phase)
        if ping is None:
            n_unmatched_ping += 1
            continue

        final_phase = utter_phase
        if utter_phase == "INBOUND_UNSPECIFIED":
            final_phase = ping["phase"]
            n_backfilled += 1

        n_matched += 1
        results.append({
            "day": day_bare, "wav_file": wav_file, "seg_start_s": seg_start,
            "asr_text": asr_text, "confidence": confidence,
            "utterance_phase": final_phase,
            "matched_tail": ping["Tail"], "matched_range_nm": ping["range_nm"],
            "matched_agl": ping["agl"], "matched_adsb_phase": ping["phase"],
            "active_runway": ping["active_runway"],
            "synthetic_reference": ping["synthetic_reference"],
            "time_diff_s": round(ping["time_diff_s"], 2),
            "phase_agrees": final_phase == ping["phase"],
        })

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"STEP 4c COMPLETE")
print(f"  Utterances total          : {n_total}")
print(f"  No phase keyword match    : {n_unmatched_phase}")
print(f"  No ADS-B ping match       : {n_unmatched_ping}")
print(f"  Inbound distance backfilled: {n_backfilled}")
print(f"  Successfully joined       : {n_matched}  ({100*n_matched/n_total:.1f}%)")

if len(df_out) == 0:
    print("\n⚠ ZERO utterances matched — check day/path wiring before proceeding")
else:
    print(f"\n  Utterance phase distribution:")
    print(df_out["utterance_phase"].value_counts())
    print(f"\n  Phase agreement (utterance keyword == ADS-B ground truth):")
    print(df_out["phase_agrees"].value_counts())
    print(f"\n  Mean time_diff_s: {df_out['time_diff_s'].mean():.1f}s")
    print(f"\n  Saved → {OUT_CSV}")
    print(f"\n  Sample rows:")
    print(df_out[["asr_text","utterance_phase","matched_adsb_phase",
                  "active_runway","synthetic_reference"]].head(8).to_string())