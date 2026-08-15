# PCAS: Pilot Communication Assistant System

> **A Novel Pilot Communication Assistant System (PCAS) for Non-Towered Airports**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Dataset: TartanAviation](https://img.shields.io/badge/Dataset-TartanAviation-green.svg)](https://github.com/castacks/tartanaviation)

---

## Overview

General aviation operations at non-towered airports depend entirely on voluntary pilot self-announcements over the Common Traffic Advisory Frequency (CTAF), with no centralized authority to detect or correct miscommunication. PCAS is the first AI-assisted communication verification system designed specifically for this setting.

PCAS fuses real-time **ADS-B surveillance data**, **surface weather (METAR)**, and **airport geometry** with a domain-adapted **Automatic Speech Recognition (ASR)** model and **Large Language Model (LLM) post-correction** to verify and correct pilot radio callouts before broadcast.

The system also introduces **ASER (Aviation Semantic Error Rate)**, a new annotation-free, entity-weighted evaluation metric that penalizes safety-critical transcription errors — particularly incorrect runway identification — more heavily than standard Word Error Rate.

---

## Key Results

Evaluated on **1,952 traffic-pattern utterances** from the [TartanAviation](https://github.com/castacks/tartanaviation) KBTP corpus (October 2020):

| Condition | WER ↓ | ASER ↓ |
|---|---|---|
| ASR only (baseline) | 107.2% | 0.560 |
| + LLM, no ADS-B context | 100.3% | 0.696 |
| + LLM + ADS-B (PCAS) | **97.7%** | **0.447** |

- **20.2% relative ASER reduction** over uncorrected ASR baseline
- LLM correction **without** ADS-B grounding degrades accuracy — surveillance context is necessary, not optional
- Verification pass rate **64.8%** under accurate context vs **17.8%** under mismatched context (3.6× difference)

---

## System Architecture

```
ADS-B + Weather + Airport/Runway Geometry
           │
           ▼
  Synthetic Reference Generator
  (Phase Classification + 5W Callout)
           │
           ├──────────────────────────────┐
           │                              │
  CTAF Audio ──► VAD ──► ASR             │
                          │               │
                          ▼               ▼
               Confidence-Driven Fast-Bypass Module
                    │              │
               [PASS]          [FAIL]
                    │              │
                    │         LLM Post-Correction
                    │         (Mistral-7B-Instruct-v0.3)
                    │              │
                    └──────┬───────┘
                           ▼
                  Verification Module
                  (Runway, Callsign, ASER checks)
                    │              │
               [PASS]          [FAIL]
                    │              │
            Verified Text     Original Audio
              Output           Fallback
```

---

## Pipeline Scripts

The full pipeline is implemented across the following scripts, organized in execution order:

| Script | Description |
|---|---|
| `download_day_data.py` | Downloads TartanAviation audio and ADS-B data from CMU S3 |
| `notebooks/08_step3_corpus_asr.py` | VAD + Whisper ASR on corpus audio, traffic-pattern filtering |
| `notebooks/09_step4b_synthetic_references.py` | Phase classification + synthetic reference generation |
| `notebooks/10_step4c_join_utterances.py` | Temporal join of ASR utterances to ADS-B synthetic references |
| `notebooks/11_step4d_ablation_no_context.py` | LLM correction ablation (no ADS-B context condition) |
| `notebooks/12_step4d_pipeline.py` | Full PCAS pipeline: fast-bypass + LLM + verification |
| `notebooks/06_compute_aser_three_conditions.py` | ASER computation across all three conditions |
| `notebooks/07_compute_wer_three_conditions.py` | WER computation across all three conditions |

Earlier single-day analysis (October 22, 2020 test set):

| Notebook | Description |
|---|---|
| `notebooks/03_asr_baseline.ipynb` | ASR baseline evaluation, three-model comparison |
| `notebooks/04_gold_transcript_evaluation.ipynb` | Gold transcript alignment and evaluation |
| `notebooks/05_asr_llm_transcript.ipynb` | LLM post-correction (Approach 1) |

---

## ASER: Aviation Semantic Error Rate

ASER is a composite metric combining embedding-level semantic similarity with a weighted entity error rate:

$$\text{ASER}(h, r) = \alpha \cdot \text{SemWER}(h, r) + (1-\alpha) \cdot \text{EER}(h, r)$$

where SemWER = 1 − cos(φ(h), φ(r)) using `all-MiniLM-L6-v2` sentence embeddings, and EER weights safety-critical entities by operational consequence:

| Entity | Weight | Rationale |
|---|---|---|
| Runway identifier | 3.0 | Direct cause of runway incursions |
| Position descriptor | 1.5 | Creates sequencing conflicts |
| Airport name | 1.0 | Recoverable from context |
| Callsign | 0.5 | Recoverable from context |

Mixing coefficient α = 0.4 prioritizes entity correctness over holistic semantic similarity.

---

## Dataset

This project uses the **[TartanAviation](https://github.com/castacks/tartanaviation)** dataset (Patrikar et al., 2025), the only publicly available corpus with co-located CTAF audio and ADS-B trajectory data from a non-towered airport.

- **Airport:** Butler County Airport, KBTP, western Pennsylvania
- **Evaluation period:** October 2020 (16 days)
- **Audio:** 533 WAV files across 16 days
- **ADS-B:** 570,929 classified pings
- **Utterances:** 1,952 traffic-pattern segments joined to synthetic references

**Note:** Raw audio and ADS-B data are not included in this repository due to size. Download instructions are in `download_day_data.py`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Shreyas-Vijayakumar/PCAS.git
cd PCAS

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

For HPC environments (tested on Iowa State University Nova cluster with A100 GPU):

```bash
pip install -r requirements_hpc.txt --no-cache-dir
```

---

## Models Used

| Component | Model | Source |
|---|---|---|
| ASR | `jlvdoorn/whisper-small.en-atco2-asr` | HuggingFace |
| LLM Post-Correction | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace |
| Sentence Embeddings | `all-MiniLM-L6-v2` | sentence-transformers |

---

## Figures

Result figures are in the `outputs/` directory:

| Figure | Description |
|---|---|
| `fig_pipeline_funnel.pdf` | PCAS pipeline outcomes for the 1,952-utterance corpus |
| `fig_aser_comparison.pdf` | ASER across three correction conditions |
| `fig_phase_agreement.pdf` | Verification pass rate by ADS-B context alignment quality |
| `fig_context_quality_breakdown.pdf` | Phase agreement breakdown |

---

## Citation

If you use this code or the ASER metric in your research, please cite:

<!-- ```bibtex
 @inproceedings{vijayakumar2026pcas,
  author    = {Bangalore Vijayakumar, Shreyas and Somani, Arun K.},
  title     = {A Novel Pilot Communication Assistant System ({PCAS})
               for Non-Towered Airports},
  booktitle = {2026 IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
  note      = {Under review}
 }
 ```-->

---

## Acknowledgments

This research was conducted at Iowa State University. We thank the CMU AirLab for the TartanAviation dataset.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

The TartanAviation dataset is subject to its own terms of use; see the [TartanAviation repository](https://github.com/castacks/tartanaviation) for details.
