# Vision-driven supernumerary robotic limb for bimanual precision tasks

Code, hardware and analysis accompanying the MSc thesis *Vision-driven
Supernumerary Robotic Limbs: Do many hands make light work?* (Imperial College
London, Department of Bioengineering, 2026).

A 6-DoF Unitree Z1 infers operator intent from tracked handheld tools and
reorients a workpiece, evaluated against unaided operation (C0) and a passively
held workpiece (C1) on a soldering-equivalent bimanual task (n = 12,
324 trials).

## Layout

| Path | Contents |
|---|---|
| `analysis/` | Statistical pipeline producing all reported results |
| `Experimental_code/` | C2 controller, perception, session runner |
| `hardware/` | CAD, button-pad firmware, calibration scripts and targets |
| `sdk/` | Unitree Z1 SDK, modified for LSL velocity streams |
| `results/` | Generated figures and tables |

Each folder has its own README.

## Reproducing the reported results

Every statistic, figure and table in the thesis comes from
`analysis/all_results.py`. Three extraction scripts feed it:

```bash
# 1. Per-trial performance metrics
python3 analysis/run_analysis.py --recordings recordings --out analysis_out

# 2. Singleton stream, face uniformity, learning effects
python3 analysis/analyse_buttons.py --rec recordings --out analysis_out

# 3. C2 telemetry
python3 analysis/orient_telemetry_analysis2.py --csv "recordings/*.csv" --out c2_evidence

# 4. All reported results
python3 analysis/all_results.py \
  --trials     analysis_out/per_trial_metrics.csv \
  --singletons analysis_out/singleton_by_trial.csv \
  --telemetry  c2_evidence/per_c2_trial.csv \
  --workbook   SUS_NASA-TLX.xlsx \
  --raw        "recordings/*.csv" \
  --outdir     results_outfinal12sus
```

Corrections are Bonferroni within hypothesis family; test families are fixed by
measurement type rather than selected from normality checks. The workload
equivalence test (TOST, ±10 raw NASA-TLX points) uses the same paired-*t*
procedure on the same participant-level differences as the primary contrast.

`analysis/deprecated/thesis_results.py` is an earlier permutation-based
pipeline with Holm correction. It returns different p-values for several
endpoints and is **not** the source of any reported value; it is retained for
transparency.

## Data

Participant data are not published. The study was approved by the Imperial
College Research Ethics Committee (approval number pending); participants consented
to anonymised storage on an encrypted Imperial College server, not to public
release. Requests are subject to ethics approval.

The pipeline above therefore cannot be re-executed on the original data. The
scripts are provided so the analysis can be inspected and reused.

## Licence

GNU GPL v3 (see `LICENSE`). `sdk/` derives from
[Aightech/z1_simple](https://github.com/Aightech/z1_simple) by A. Devillard,
also GPL v3; modifications are noted in `sdk/NOTICE`.

## Attribution

DodecaPen tracking follows Wu et al., *DodecaPen: Accurate 6DoF Tracking of a
Passive Stylus*, UIST 2017. 