"""
Step 3 — Run ASR on all 481 corpus WAVs, filter traffic pattern segments.

Pipeline per WAV:
  1. Load audio (16kHz mono)
  2. VAD  → speech segments
  3. ASR  → transcription (whisper-small-atco2)
  4. Hallucination filter
  5. Keyword filter → keep only traffic pattern segments
  6. Save to JSONL

Output:
  outputs/corpus_segments_raw.jsonl    ← all clean segments (before keyword filter)
  outputs/corpus_traffic_pattern.jsonl ← keyword-filtered traffic pattern only
  outputs/corpus_summary.csv           ← per-file stats

Estimated time: 45-90 minutes on A100
"""

import json
import os
import re
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
XCAS        = Path("/work/arun/shreyasv/xcas-ga")
CORPUS_DIR  = XCAS / "data" / "corpus_audio"
OUT_DIR     = XCAS / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW     = OUT_DIR / "corpus_segments_raw.jsonl"
OUT_TRAFFIC = OUT_DIR / "corpus_traffic_pattern.jsonl"
OUT_SUMMARY = OUT_DIR / "corpus_summary.csv"

os.environ["HF_HOME"] = str(XCAS / "cache/hf")

# ── Traffic pattern keywords ───────────────────────────────────────────────────
# Segment is traffic-pattern-relevant if it contains ANY of these
TRAFFIC_KEYWORDS = {
    # Pattern legs
    "downwind", "crosswind", "upwind", "base", "final",
    # Entries
    "inbound", "straight-in", "teardrop", "entering", "pattern",
    # Movements
    "departing", "departure", "rolling", "cleared",
    "touch", "landing", "takeoff", "go around", "overflight",
    "overflying", "staying",
}

# Hallucination patterns (same as test set pipeline)
HALLUCINATION_RE = re.compile(
    "|".join([
        r"thank you (so much )?for (joining|watching)",
        r"(okay[,\.]?\s*){3,}",
        r"(\.\s*){4,}",
        r"^you\.?$",
        r"^\s*(the|a|an)\s*$",
        r"(the next one[^\n]*){2,}",
    ]),
    re.IGNORECASE | re.DOTALL
)

ATC_KEYWORDS = {
    "runway", "traffic", "butler", "inbound", "outbound", "departing",
    "final", "downwind", "base", "cleared", "takeoff", "landing",
    "miles", "north", "south", "east", "west", "cessna", "piper",
    "november", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "oscar",
    "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
    "whiskey", "xray", "yankee", "zulu",
}

def is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True
    if HALLUCINATION_RE.search(text.strip()):
        return True
    words = text.strip().lower().split()
    if len(words) <= 2 and not any(kw in text.lower() for kw in ATC_KEYWORDS):
        return True
    return False

def is_traffic_pattern(text: str) -> bool:
    """Return True if segment contains any traffic pattern keyword."""
    t = text.lower()
    return any(kw in t for kw in TRAFFIC_KEYWORDS)

# ── VAD ────────────────────────────────────────────────────────────────────────
def vad_segments(audio: np.ndarray, sr: int = 16000,
                 snr_db: float = 8.0,
                 min_dur: float = 1.5,
                 buffer: float = 0.3) -> list[dict]:
    """Energy-based VAD. Returns list of {start, end, duration} dicts."""
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_len,
                               hop_length=hop_len)[0]
    noise_floor = np.percentile(rms, 30)
    threshold   = noise_floor * (10 ** (snr_db / 20))

    in_speech = False
    segs, seg_start = [], 0.0
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_len
    )

    for i, (t, r) in enumerate(zip(times, rms)):
        if not in_speech and r > threshold:
            in_speech  = True
            seg_start  = max(0.0, t - buffer)
        elif in_speech and r <= threshold:
            in_speech  = False
            seg_end    = min(len(audio)/sr, t + buffer)
            dur        = seg_end - seg_start
            if dur >= min_dur:
                segs.append({"start": round(seg_start, 3),
                             "end":   round(seg_end,   3),
                             "duration": round(dur, 3)})

    if in_speech:
        seg_end = len(audio) / sr
        dur     = seg_end - seg_start
        if dur >= min_dur:
            segs.append({"start": round(seg_start, 3),
                         "end":   round(seg_end,   3),
                         "duration": round(dur, 3)})
    return segs

# ── Load model ─────────────────────────────────────────────────────────────────
def load_whisper():
    model_id = "jlvdoorn/whisper-small.en-atco2-asr"
    print(f"Loading {model_id} ...")
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype         = torch.float16,
        attn_implementation = "eager",
    ).to("cuda").eval()
    print(f"Loaded in {time.time()-t0:.1f}s  "
          f"| VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return processor, model

# ── ASR inference ──────────────────────────────────────────────────────────────
def transcribe_segment(audio_slice: np.ndarray,
                        processor, model) -> tuple[str, float]:
    """
    Transcribe one audio segment.
    Returns (text, mean_log_prob_confidence).
    """
    inputs = processor(
        audio_slice, sampling_rate=16000, return_tensors="pt"
    )
    feats = inputs.input_features.to("cuda", dtype=torch.float16)

    with torch.no_grad():
        out = model.generate(
            feats,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=80,
            condition_on_prev_tokens=False,
        )

    text = processor.decode(
        out.sequences[0], skip_special_tokens=True
    ).strip()

    # Mean token log-probability as confidence
    if out.scores:
        log_probs = [
            torch.log_softmax(s, dim=-1)
                  .max(dim=-1).values.item()
            for s in out.scores
        ]
        confidence = float(np.mean(log_probs)) if log_probs else -999.0
    else:
        confidence = -999.0

    return text, confidence

# ── Process one WAV file ───────────────────────────────────────────────────────
def process_wav(wav_path: Path, processor, model) -> dict:
    """Full pipeline for one WAV file. Returns result dict."""
    try:
        audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    except Exception as e:
        return {"wav_file": wav_path.name, "day": wav_path.parent.name,
                "error": str(e), "segments": []}

    segs = vad_segments(audio)
    results = []

    for seg in segs:
        s0 = int(seg["start"] * 16000)
        s1 = int(seg["end"]   * 16000)
        slice_audio = audio[s0:s1]

        text, conf = transcribe_segment(slice_audio, processor, model)
        halluc     = is_hallucination(text)
        traffic    = is_traffic_pattern(text) if not halluc else False

        results.append({
            "start"           : seg["start"],
            "end"             : seg["end"],
            "duration"        : seg["duration"],
            "text"            : text,
            "confidence"      : round(conf, 4),
            "is_hallucination": halluc,
            "is_traffic_pattern": traffic,
        })

    n_clean   = sum(1 for s in results if not s["is_hallucination"])
    n_traffic = sum(1 for s in results if s["is_traffic_pattern"])

    return {
        "wav_file"      : wav_path.name,
        "day"           : wav_path.parent.name,
        "wav_path"      : str(wav_path),
        "n_segments"    : len(results),
        "n_clean"       : n_clean,
        "n_traffic"     : n_traffic,
        "segments"      : results,
    }

# ── Main loop ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert torch.cuda.is_available(), "No GPU available"
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    # Collect all corpus WAVs
    all_wavs = sorted([
        wav
        for day_dir in sorted(CORPUS_DIR.iterdir())
        if day_dir.is_dir()
        for wav in sorted(day_dir.glob("*.wav"))
    ])
    print(f"Corpus WAVs found : {len(all_wavs)} across "
          f"{len(list(CORPUS_DIR.iterdir()))} days")

    # Load model once
    processor, model = load_whisper()

    # Track already-processed WAVs for resumability
    processed = set()
    if OUT_RAW.exists():
        with open(OUT_RAW) as f:
            for line in f:
                r = json.loads(line)
                processed.add(r["wav_file"] + "|" + r["day"])
        print(f"Resuming — {len(processed)} WAVs already processed")

    # Open output files in append mode
    f_raw     = open(OUT_RAW,     "a")
    f_traffic = open(OUT_TRAFFIC, "a")

    summary_rows = []
    t_start      = time.time()
    n_done       = 0

    for i, wav_path in enumerate(all_wavs):
        key = wav_path.name + "|" + wav_path.parent.name
        if key in processed:
            continue

        t_wav = time.time()
        result = process_wav(wav_path, processor, model)
        elapsed = time.time() - t_wav

        # Write raw record
        f_raw.write(json.dumps(result) + "\n")
        f_raw.flush()

        # Write traffic pattern segments separately
        for seg in result.get("segments", []):
            if seg["is_traffic_pattern"]:
                record = {
                    "day"        : result["day"],
                    "wav_file"   : result["wav_file"],
                    "start"      : seg["start"],
                    "end"        : seg["end"],
                    "duration"   : seg["duration"],
                    "text"       : seg["text"],
                    "confidence" : seg["confidence"],
                }
                f_traffic.write(json.dumps(record) + "\n")
        f_traffic.flush()

        summary_rows.append({
            "day"       : result["day"],
            "wav_file"  : result["wav_file"],
            "n_segs"    : result["n_segments"],
            "n_clean"   : result["n_clean"],
            "n_traffic" : result["n_traffic"],
            "elapsed_s" : round(elapsed, 2),
        })

        n_done += 1
        total_elapsed = time.time() - t_start
        remaining     = len(all_wavs) - i - 1
        rate          = n_done / total_elapsed
        eta_min       = remaining / rate / 60 if rate > 0 else 0

        print(f"  [{i+1:3d}/{len(all_wavs)}] {wav_path.parent.name}/"
              f"{wav_path.name:<12}  "
              f"segs={result['n_segments']:3d}  "
              f"traffic={result['n_traffic']:3d}  "
              f"t={elapsed:.1f}s  "
              f"ETA={eta_min:.0f}m")

    f_raw.close()
    f_traffic.close()

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_SUMMARY, index=False)

    # Final report
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"STEP 3 COMPLETE")
    print(f"{'='*60}")
    print(f"  WAVs processed      : {n_done}")
    print(f"  Total segments      : {df['n_segs'].sum()}")
    print(f"  Clean segments      : {df['n_clean'].sum()}")
    print(f"  Traffic pattern segs: {df['n_traffic'].sum()}")
    print(f"  Saved raw           : {OUT_RAW}")
    print(f"  Saved traffic only  : {OUT_TRAFFIC}")
    print(f"  Saved summary CSV   : {OUT_SUMMARY}")
    print(f"  Total time          : {total_time/60:.1f} minutes")
    print(f"  Output files ready for Step 4")