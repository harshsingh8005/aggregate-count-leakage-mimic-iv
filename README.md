# aggregate-count-leakage-mimic-iv

Code and aggregate outputs for:

> Harsh Singh. **A Count Is Not a Code: Aggregate Label Leakage That Survives
> Per-Code De-Leaking on MIMIC-IV.** SSRN preprint, 2026.
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7676958

## What the paper shows

Diagnosis codes in hospital records are assigned at discharge. A common fix for
label leakage removes the current admission's codes as features. The number of
distinct diagnosis codes on the admission keeps no code's identity, so a check
that looks at codes one at a time never sees it. On first ICU stays in MIMIC-IV
v3.1, that count predicts in-hospital death on its own and adds to a model
already cleaned of code features, although it is not available at a 24-hour
prediction time.

## What is in this repository

```
audit/            the scripts behind the first version of the paper
  cohort_build.py               builds the 85,242-admission first ICU stay cohort
  arms.py                       feature sets (a) to (d), parameterised encoder
  analysis.py                   feature sets, isolation test, Spearman rho, storage time
  task4_task5b.py               comorbidity index coverage, storage-time exclusions
  comparators.py                single-feature rank AUROC: count, stay length, age
  mechanism.py                  stay-length strata, cap mass
  icd.py                        ICD-9 and ICD-10 split
  anchor_and_stratification.py  24-hour cohort, conditional AUROC, recoded stay length
  isolation_sweep.py            laboratory features scored on their own
  vitals_sweep.py               vital-sign features scored on their own
  categorical_sweep.py          admission categoricals scored on their own
  sweep_intervals.py            bootstrap intervals, count and laboratory features
  vitals_intervals.py           bootstrap intervals, vital signs
  resp_itemid_split.py          the respiratory-rate presence example
  comorbidity_isolation.py      the comorbidity index scored on its own
  stay_length_controls.py       stay-length controls and the chronic-code checks
  horizon_and_anchor.py         fixed-horizon labels; stay length at the anchor
  arms_at_anchor.py             Table 1 feature sets on the 24-hour cohort
  paired_delta.py               paired bootstrap of the differences between sets
revision/         scripts added for the current version
  extended_checks.py            cohort flow, one time origin (t0), horizons and
                                stay length from t0, chronic-code checks and
                                ICD split on the 24-hour cohort, other
                                admission-level counts (Table 2)
  robustness_checks.py          logistic regression, per-set tuning and two
                                calendar-time splits. NOT YET RUN; no output
                                is shipped and the paper reports no number from it
reproducer_src/src/   the pipeline package the scripts import, unchanged from
                      the audited reproducer
outputs/          what the scripts wrote (aggregate statistics only)
  revision/       outputs of extended_checks.py
NUMBERS.md        every number in the paper and the file it comes from
```

## Data access

The data are not included and cannot be. MIMIC-IV is available only to
credentialed researchers through PhysioNet, under a data use agreement that
forbids redistribution. To run the code you need:

1. A PhysioNet account with credentialed access, including completed human
   subjects research training.
2. A signed data use agreement for MIMIC-IV v3.1
   (https://physionet.org/content/mimiciv/3.1/).
3. Optionally, MIMIC-IV-Note 2.2 for the radiology report count in Table 2.

## Running it

Python 3.11. Install the packages in `requirements.txt`.

```bash
export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
export MIMIC_IV_NOTE_PATH=/path/to/physionet.org/files/mimic-iv-note/2.2   # optional
export DUCKDB_THREADS=4 DUCKDB_MEMORY=6GB    # defaults; raise on a larger machine

python audit/analysis.py                  # feature sets, isolation test, storage time (about 38 min at 14 threads)
python audit/task4_task5b.py
python audit/anchor_and_stratification.py
python audit/isolation_sweep.py
python audit/vitals_sweep.py
python audit/categorical_sweep.py
python audit/sweep_intervals.py
python audit/vitals_intervals.py
python audit/resp_itemid_split.py
python audit/mechanism.py
python audit/comparators.py
python audit/icd.py
python audit/comorbidity_isolation.py
python audit/stay_length_controls.py
python audit/horizon_and_anchor.py
python audit/arms_at_anchor.py            # slow: reads chartevents and labevents
python audit/paired_delta.py              # slow: reads chartevents and labevents
python revision/extended_checks.py        # about 10 min
```

On Windows, use `set NAME=value` instead of `export`.

Each script writes to `outputs/` (or `outputs/revision/`) under the file names
listed in `NUMBERS.md`. No script contains an absolute path.

### Checks before trusting a re-run

* `revision/extended_checks.py` first recomputes 27 values from the first
  version of the paper with the original definitions and prints PASS or FAIL
  for each. All 27 passed on the shipped run
  (`outputs/revision/extended_checks_console.log`). Do not quote a new number
  if a check fails.
* `audit/cohort_build.py` and `audit/arms.py` can be checked without
  credentialed access against the open MIMIC-IV demo (100 patients): set
  `MIMIC_IV_DEMO_PATH` and run `python audit/cohort_build.py`. This confirms
  that the reimplemented cohort and encoder match the reproducer's own loader
  and feature builder column for column.

## Privacy

Every script writes aggregate statistics only: counts, means, quantiles and
metrics. No cohort list, identifier, timestamp or row-level value is written or
shipped. In the public copy of `outputs/revision/extended_checks.json` and its
console log, cells below 10 admissions are suppressed, together with any cell
that would reveal one by subtraction.

## Provenance of the revision scripts

`revision/extended_checks.py` is the script that produced
`outputs/revision/extended_checks.json`, with three kinds of change made for
this repository and none to the computation:

1. file paths now point at this repository's layout;
2. the script and its output were renamed (originally `rebuttal_checks.py` and
   `rebuttal_checks.json`; the run copy had MD5
   `ad66724ac3e0d82a12876e1d31e39539`);
3. docstrings and comments that referred to an earlier review were reworded.

The same three kinds of change were made to `revision/robustness_checks.py`.

## Known limits

* All results are from one database and one hospital.
* The feature set comparison uses one fixed LightGBM configuration and one
  patient-grouped split. `revision/robustness_checks.py` tests other models and
  splits but has not been run yet.
* See the paper's Discussion for the full list.

## License

MIT, see `LICENSE`. The MIMIC-IV data are governed by the PhysioNet
Credentialed Health Data License and are not covered by this license.

## Citation

See `CITATION.cff`.
