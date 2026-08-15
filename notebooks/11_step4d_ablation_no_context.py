"""
Ablation — LLM correction WITHOUT ADS-B/synthetic-reference context.
Same 1952 utterances, same Mistral model, same generation params as
the main PCAS run, but the prompt omits tail number, phase, range,
and synthetic reference entirely. This isolates the contribution of
ADS-B context injection from the LLM's general correction ability.

Input : outputs/utterance_synthetic_pairs.csv (for asr_text + reference for scoring)
Output: outputs/pcas_ablation_no_context.csv
"""

import re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

XCAS    = Path("/work/arun/shreyasv/xcas-ga")
IN_CSV  = XCAS / "outputs" / "utterance_synthetic_pairs.csv"
OUT_CSV = XCAS / "outputs" / "pcas_ablation_no_context.csv"

import os
os.environ["HF_HOME"] = str(XCAS / "cache/hf")

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

# ── No-context system prompt — generic ASR correction only ───────────────────
SYSTEM_PROMPT_NOCTX = """You are an aviation radio communication corrector for a non-towered general aviation airport.

Correct the ASR transcript to standard CTAF self-announce phraseology.
Rules:
1. Return ONLY the corrected transcript, nothing else
2. Fix likely mishearings of airport/aircraft names
3. Preserve words you are not confident are wrong
4. Never invent information absent from the input"""

def correct_no_context(asr_text):
    user_msg = f'Raw ASR transcript: "{asr_text}"\n\nCorrected transcript:'
    messages = [{"role":"system","content":SYSTEM_PROMPT_NOCTX},
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

print("\nLoading utterance pairs...")
df = pd.read_csv(IN_CSV, dtype={"active_runway": str})
print(f"  {len(df)} pairs loaded")

results = []
t0 = time.time()
for i, row in df.iterrows():
    corrected = correct_no_context(row["asr_text"])
    results.append({
        "day": row["day"], "wav_file": row["wav_file"],
        "asr_text": row["asr_text"],
        "synthetic_reference": row["synthetic_reference"],
        "active_runway": row["active_runway"],
        "llm_output_noctx": corrected,
    })
    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed/(i+1)*(len(df)-i-1)
        print(f"  [{i+1}/{len(df)}]  {elapsed:.0f}s elapsed  ETA {eta/60:.1f}min")

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nAblation complete ({(time.time()-t0)/60:.1f} min). Saved → {OUT_CSV}")