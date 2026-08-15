"""
Step 4b — Phase classification + synthetic reference generation, all 16 days.
Consumes the runway timeline logic from step4a.
Output: outputs/synthetic_references_all_days.csv
        one row per classified ADS-B ping (AGL<3000, one of 8 phases)
        with phase, active_runway, and the 5W synthetic reference string.
"""

import re, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step4a_runway_inference import (
    load_day_adsb, load_weather, build_runway_timeline,
    KBTP_LAT, KBTP_LON, RUNWAY_HDG, bearing_from_airport,
)

warnings.filterwarnings("ignore")

XCAS      = Path("/work/arun/shreyasv/xcas-ga")
OUT_CSV   = XCAS / "outputs" / "synthetic_references_all_days.csv"

GROUND_ELEV_FT = 1248.4
AGL_CEILING    = 3000          # hard pre-filter: drop pings >= this AGL
HDG_TOL        = 10            # heading tolerance, degrees

ALL_DAYS = [
    "10-01-20","10-02-20","10-03-20","10-10-20","10-11-20","10-12-20",
    "10-13-20","10-21-20","10-22-20","10-23-20","10-24-20","10-25-20",
    "10-27-20","10-30-20","10-31-20",
]   # 10-26-20 has no ADS-B (confirmed upstream); 15 days total

PHONETIC = {
    "A":"Alpha","B":"Bravo","C":"Charlie","D":"Delta","E":"Echo","F":"Foxtrot",
    "G":"Golf","H":"Hotel","I":"India","J":"Juliet","K":"Kilo","L":"Lima",
    "M":"Mike","N":"November","O":"Oscar","P":"Papa","Q":"Quebec","R":"Romeo",
    "S":"Sierra","T":"Tango","U":"Uniform","V":"Victor","W":"Whiskey",
    "X":"Xray","Y":"Yankee","Z":"Zulu",
}

def tail_to_phonetic(tail):
    if not isinstance(tail, str) or not tail.strip():
        return "Unknown Traffic"
    out = []
    for ch in tail.upper().strip():
        if ch.isalpha():   out.append(PHONETIC.get(ch, ch))
        elif ch.isdigit(): out.append(ch)
    return " ".join(out)

def bearing_to_compass(b):
    dirs = [(22.5,"North"),(67.5,"Northeast"),(112.5,"East"),(157.5,"Southeast"),
            (202.5,"South"),(247.5,"Southwest"),(292.5,"West"),(337.5,"Northwest"),
            (360.0,"North")]
    b %= 360
    for thr, name in dirs:
        if b <= thr: return name
    return "North"

def round_callout_distance(nm):
    if nm >= 8:   return 10
    elif nm >= 4: return 5
    else:         return 3

def hdiff(h, target):
    return abs((h - target + 180) % 360 - 180)

# ── Phase classifier ────────────────────────────────────────────────────────────
def classify_phase(agl, speed, hdg, rng_nm, rwy_hdg):
    """8 target phases + OTHER. AGL already computed (MSL-1248.4)."""
    if pd.isna(hdg) or pd.isna(rng_nm):
        return "OTHER"

    # Step 1 — ground
    if agl < 200 and (pd.isna(speed) or speed < 50):
        return "OTHER"

    # Step 2 — heading-axis legs (±10°)
    if rng_nm <= 3 and 200 <= agl <= 600 and hdiff(hdg, rwy_hdg) <= HDG_TOL:
        return "FINAL"
    if rng_nm <= 3 and 600 <= agl <= 800 and hdiff(hdg, rwy_hdg+270) <= HDG_TOL:
        return "BASE"
    if rng_nm <= 2 and 600 <= agl <= 1000 and hdiff(hdg, rwy_hdg+90) <= HDG_TOL:
        return "CROSSWIND"
    if 0.5 <= rng_nm <= 2.5 and 800 <= agl <= 1200 and hdiff(hdg, rwy_hdg+180) <= HDG_TOL:
        return "DOWNWIND"

    # Step 3 — departing
    # (climb trend evaluated by caller via alt_delta_smooth; passed as speed-sign hack avoided)
    # handled outside; placeholder returns OTHER here, real check in apply loop

    # Step 4 — pattern entry
    if rng_nm <= 3 and 1200 <= agl <= 2000:
        return "PATTERN_ENTRY"

    # Step 5 — inbound (range-gated)
    if 10 <= rng_nm <= 19: return "INBOUND_10NM"
    if 5  <= rng_nm < 10:  return "INBOUND_5NM"
    if 3  <= rng_nm < 5:   return "INBOUND_3NM"

    return "OTHER"

# ── Synthetic reference (5W) ──────────────────────────────────────────────────
def make_reference(phase, tail, rng_nm, bearing, rwy):
    if phase == "OTHER":
        return None
    who   = tail_to_phonetic(tail)
    direction = bearing_to_compass(bearing)
    dist  = round_callout_distance(rng_nm)
    A = "Butler Traffic"
    if phase == "INBOUND_10NM" or phase == "INBOUND_5NM" or phase == "INBOUND_3NM":
        return f"{A} {who} {dist} miles {direction} inbound runway {rwy} {A}"
    if phase == "PATTERN_ENTRY":
        return f"{A} {who} entering downwind runway {rwy} {A}"
    if phase == "CROSSWIND":
        return f"{A} {who} crosswind runway {rwy} {A}"
    if phase == "DOWNWIND":
        return f"{A} {who} downwind runway {rwy} {A}"
    if phase == "BASE":
        return f"{A} {who} base runway {rwy} {A}"
    if phase == "FINAL":
        return f"{A} {who} final runway {rwy} {A}"
    if phase == "DEPARTING":
        return f"{A} {who} departing runway {rwy} {A}"
    return None

# ── Per-day processing ──────────────────────────────────────────────────────────
def process_day(day, runway_lookup):
    df = load_day_adsb(day)

    # AGL conversion + hard ceiling filter
    df["agl"] = df["Altitude"] - GROUND_ELEV_FT
    df = df[df["agl"] < AGL_CEILING].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # climb trend per aircraft (for DEPARTING)
    df = df.sort_values(["Tail","ts_aware"])
    df["alt_delta"] = df.groupby("Tail")["Altitude"].diff()
    df["alt_delta_smooth"] = (df.groupby("Tail")["alt_delta"]
                                .transform(lambda x: x.rolling(3, min_periods=1).mean()))

    # assign per-ping active runway from minute-bucketed timeline
    df["bucket"] = df["ts_aware"].dt.floor("min")
    df["active_runway"] = df["bucket"].map(runway_lookup).fillna("26")

    rows = []
    for _, r in df.iterrows():
        rwy = r["active_runway"]
        rwy_hdg = RUNWAY_HDG[rwy]
        phase = classify_phase(r["agl"], r["Speed"], r["Heading"],
                               r["range_nm"], rwy_hdg)
        # DEPARTING override (needs climb trend, checked here)
        if phase == "OTHER" and r["range_nm"] <= 1.5 \
           and pd.notna(r["alt_delta_smooth"]) and r["alt_delta_smooth"] > 20 \
           and hdiff(r["Heading"], rwy_hdg) <= HDG_TOL:
            phase = "DEPARTING"

        if phase == "OTHER":
            continue

        ref = make_reference(phase, r["Tail"], r["range_nm"],
                             r["bearing_deg"], rwy)
        rows.append({
            "day": day, "timestamp": r["timestamp"], "ts_aware": r["ts_aware"],
            "Tail": r["Tail"], "range_nm": round(r["range_nm"],3),
            "bearing_deg": round(r["bearing_deg"],1), "agl": round(r["agl"],1),
            "heading": r["Heading"], "phase": phase,
            "active_runway": rwy, "synthetic_reference": ref,
        })
    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Step 4b — Synthetic reference generation, 15 days")
    all_out = []
    for day in ALL_DAYS:
        tl = build_runway_timeline(day)
        runway_lookup = dict(zip(tl["bucket_time"], tl["active_runway"]))
        day_df = process_day(day, runway_lookup)
        n = len(day_df)
        dist = day_df["phase"].value_counts().to_dict() if n else {}
        print(f"  {day}: {n:5d} classified pings  {dist}")
        if n: all_out.append(day_df)

    df_all = pd.concat(all_out, ignore_index=True)
    df_all["active_runway"] = df_all["active_runway"].astype(str).str.zfill(2)
    df_all.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"STEP 4b COMPLETE")
    print(f"  Total classified pings : {len(df_all)}")
    print(f"  Days                   : {df_all['day'].nunique()}")
    print(f"\n  Phase distribution (all days):")
    print(df_all["phase"].value_counts())
    print(f"\n  Runway distribution:")
    print(df_all["active_runway"].value_counts())
    print(f"\n  Saved → {OUT_CSV}")
    print(f"\n  Sample synthetic references:")
    for ph in df_all["phase"].unique():
        ex = df_all[df_all["phase"]==ph].iloc[0]
        print(f"    [{ph}] {ex['synthetic_reference']}")