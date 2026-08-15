"""
Compute WER for all three conditions on the full-October corpus,
referenced against the synthetic reference (consistent with how
ASER has been computed throughout this pipeline).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

XCAS = Path("/work/arun/shreyasv/xcas-ga")

def normalize(text):
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return t.split()

def wer(hyp, ref):
    h, r = normalize(hyp), normalize(ref)
    if len(r) == 0:
        return 1.0 if len(h) > 0 else 0.0
    # standard Levenshtein on word sequences
    d = np.zeros((len(r)+1, len(h)+1))
    d[:,0] = np.arange(len(r)+1)
    d[0,:] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)
    return d[len(r),len(h)] / len(r)

print("Loading main pipeline results (v3) and ablation results...")
df_main = pd.read_csv(XCAS/"outputs"/"pcas_pipeline_results_v3.csv")
df_abl  = pd.read_csv(XCAS/"outputs"/"pcas_ablation_no_context.csv")

# Align ablation rows to main rows (same order, same source file)
assert len(df_main) == len(df_abl), "Row count mismatch — check both files"

print("\nComputing WER — ASR only (baseline)...")
wer_asr = df_main.apply(lambda r: wer(r["asr_text"], r["synthetic_reference"]), axis=1)

print("Computing WER — LLM, no ADS-B context (ablation)...")
wer_noctx = df_abl.apply(lambda r: wer(r["llm_output_noctx"], r["synthetic_reference"]), axis=1)

print("Computing WER — full PCAS (bypass output OR LLM-corrected, using final_output_v2)...")
def get_pcas_text(row):
    if row["path_v2"] == "fast_bypass":
        return row["asr_text"]
    if str(row["verify_status_v2"]).startswith("PASS"):
        return row["llm_output"]
    return row["asr_text"]   # verify-fail falls back to raw ASR text for WER purposes
                              # (audio fallback has no text, so we score the raw ASR
                              # transcript as the best available text proxy)
df_main["pcas_text"] = df_main.apply(get_pcas_text, axis=1)
wer_pcas = df_main.apply(lambda r: wer(r["pcas_text"], r["synthetic_reference"]), axis=1)

print(f"\n{'='*60}")
print(f"WER RESULTS — full October corpus (n={len(df_main)})")
print(f"{'='*60}")
print(f"  ASR only (baseline)        : {wer_asr.mean():.3f}")
print(f"  + LLM, no ADS-B context    : {wer_noctx.mean():.3f}")
print(f"  + LLM + ADS-B (PCAS)       : {wer_pcas.mean():.3f}")

print(f"\nFor cross-check, mean ASER on same three conditions:")
print(f"  ASR only       : {df_main['aser_fastbypass_v2'].mean():.3f}  (fastbypass-weight ASER, all rows)")
print(f"  + LLM + ADS-B  : full ASER already computed = 0.177 (verify-pass only, from earlier)")