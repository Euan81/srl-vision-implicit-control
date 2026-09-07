# Analysis

Analysis code for the MSc thesis *Vision-driven Supernumerary Robotic Limbs:
Do many hands make light work?* (Imperial College London, Bioengineering, 2026).

## Pipeline

Three extraction scripts feed one statistics script. Every number, figure and
table reported in the thesis comes from the final step.

```bash
# 1. Per-trial performance metrics
python3 run_analysis.py --recordings recordings --out analysis_out12participants

# 2. Singleton (one-handed press) stream, face uniformity, learning effects
python3 analyse_buttons.py --rec recordings --out analysis_out12participants

# 3. C2 telemetry (tracking, attenuation, pen geometry, settle time)
python3 orient_telemetry_analysis2.py --csv "recordings/*.csv" --out c2_evidence12

# 4. All reported statistics, figures and tables
python3 thesis_results3_final2.py \
  --trials     analysis_out12participants/per_trial_metrics.csv \
  --singletons analysis_out12participants/singleton_by_trial.csv \
  --telemetry  c2_evidence12/per_c2_trial.csv \
  --workbook   SUS_NASA-TLX.xlsx \
  --raw        "recordings/*.csv" \
  --outdir     results_outfinal12sus
```

Corrections are Bonferroni within hypothesis family. Test families are fixed
by measurement type in `METRIC_FAMILY`, not selected from normality checks.
The workload equivalence test (TOST, ±10 raw NASA-TLX points) uses the same
paired-*t* procedure on the same participant-level differences as the primary
contrast.

`deprecated/thesis_results.py` is an earlier permutation-based pipeline with
Holm correction. It returns different p-values for several endpoints and is
**not** the source of any reported value. It is retained for transparency.

## Files

| File | Role |
|---|---|
| `all_results.py` | Hypothesis tests, effect sizes, all reported figures and tables |
| `run_analysis.py` | Trial segmentation and per-trial performance metrics |
| `analyse_buttons.py` | Singletons, face-dependent variability, learning effects |
| `orient_telemetry_analysis.py` | C2 controller and perception telemetry |
| `srl_analysis.py`, `srl_common.py` | Shared trial discovery, metric extraction, condition constants |
| `statsfuns.py` | Bootstrap CIs, effect sizes, correction helpers |

## Requirements

Python 3.9+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Outputs

`--outdir` contains `FIGURE_MANIFEST.csv` (figure numbering),
`ANALYSIS_MANIFEST.csv` (statistical backend and correction), and a `.tex`
caption generated from the data for each figure.

## Data

Participant data are not included; see `../data/README.md`.