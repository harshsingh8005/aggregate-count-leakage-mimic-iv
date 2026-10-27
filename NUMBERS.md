# Every number in the paper, and the file it comes from

Paper: Harsh Singh, "A Count Is Not a Code: Aggregate Label Leakage That
Survives Per-Code De-Leaking on MIMIC-IV", SSRN preprint, 2026
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7676958).

All paths are under `outputs/`. "Full cohort" is n = 85,242 (all first ICU
stays). "24-hour cohort" is n = 81,052 (alive and in hospital 24 hours after
ICU admission).

## Section 3: Data and definitions

| Value | File |
|---|---|
| 24-hour cohort: 81,052 admissions, 62,421 patients, 7,659 deaths | `arms_at_anchor.json` |
| Full cohort: 85,242 admissions, 65,366 patients, 9,475 deaths (11.12%) | `analysis.log` line 2, `arms_results.json` |
| 1,767 deaths (18.6%) within 24 h of ICU admission | `anchor_and_stratification.json`, `revision/extended_checks.json` (`flow`) |

## Section 4: Results

| Value | File |
|---|---|
| Count alone, 24-hour cohort: 0.7288 [0.7231, 0.7353] | `revision/extended_checks.json` (`other_counts_24h_cohort`, reference row) |
| Count alone, full cohort: 0.6931 [0.6875, 0.6985] (model), 0.6941 (rank) | `analysis.log` line 8, `comparators.json` |
| Means 23.35 vs 17.20 codes; Figure 1 and Table D1 quantiles | `results.json` (`task3`) |
| Blood urea nitrogen 0.7006 [0.6946, 0.7058]; count CI [0.6894, 0.6995] | `isolation_sweep.json`, `sweep_intervals.json` |
| Common support n = 82,538: count 0.7125 vs 0.7006 | `isolation_sweep.json` (`head_to_head`) |
| Comorbidity index alone 0.5510; 0.6015 where non-zero | `comorbidity_isolation.json` |
| Table 1, 24-hour cohort, sets (a) to (c) | `arms_at_anchor.json` |
| Table 1, full cohort, sets (a) to (c) | `arms_results.json`, `results.json` (`task2`), `analysis.log` lines 4 to 6 |
| Paired gain +0.01289 [+0.01115, +0.01467]; AUPRC +0.00865 [+0.00615, +0.01131] | `paired_delta.json` |
| Index non-zero for 36,319 (42.6%); 0 of 43,919 first admissions non-zero; mean 1.83; max 22 | `task4_task5b.json`, `comorbidity_isolation.json` |
| Stay length from t0 (24-hour cohort): Spearman 0.51, AUROC 0.4906, strata 0.7464 / 0.7482 / 0.7472 / 0.7473, own strata 0.48 to 0.50, recoded 0.6158 to 0.7509 | `revision/extended_checks.json` (`stay_length_controls`) |
| 93.2% of decedents with a recorded death time have discharge time equal to death time | `horizon_and_anchor.json` |
| Fixed horizons from t0 (24-hour cohort): 0.6345 / 0.6810 / 0.7157 | `revision/extended_checks.json` (`fixed_horizon`) |
| Chronic codes 0.6610, other codes 0.7130, 20 random subsets 0.6460 to 0.6537 (mean 0.6497), conditional 0.5488 and 0.6853 | `revision/extended_checks.json` (`appendix_c`) |
| ICD-9 0.6965 (n 39,430) and ICD-10 0.7591 (n 41,599) on the 24-hour cohort | `revision/extended_checks.json` (`icd_version`) |
| ICD-9 0.6562 (n 41,395) and ICD-10 0.7290 (n 43,823) on the full cohort; means 15.85 and 19.81 | `icd.json` |
| Table 2, other admission-level counts | `revision/extended_checks.json` (`other_counts_24h_cohort`) |
| Storage-time rule: 2.40% lab rows, 2.27% vital rows; AUROC 0.8615 [0.8579, 0.8654], change -0.0013 | `task4_task5b.json`, `task4_task5b.log`, `results.json` (`task5`), `analysis.log` line 13 |

## Section 5 and Discussion

| Value | File |
|---|---|
| Four features above 0.65; bicarbonate 0.0021 below | `full_sweep.json`, `sweep_intervals.json` |
| Respiratory-rate item: 37.9% of admissions, 59% of deaths (full cohort) | `resp_itemid_split.json` (`itemids.224690`) |

## Appendices

| Value | File |
|---|---|
| Table A1 cohort flow (1,767; 49; 2,374; 20 with death time before ICU admission) | `revision/extended_checks.json` (`flow`) |
| Table B2 isolation sweep: laboratory features | `isolation_sweep.json`, `sweep_intervals.json` |
| Table B2: vital signs and their intervals | `vitals_sweep.json`, `vitals_intervals.json` |
| Table B2: admission categoricals | `categorical_sweep.json` |
| Table B2: all 23 features with source script | `full_sweep.json` |
| Table C1, full cohort column (20 random draws 0.6245 to 0.6331) | `revision/extended_checks.json` (`appendix_c.submitted_cohort`), `stay_length_controls.json` |
| Appendix D cap mass: 2.14% survivors, 7.25% decedents | `mechanism.json` |
| Table F1 horizons, full cohort from t0 and from hospital admission | `revision/extended_checks.json` (`fixed_horizon`), `horizon_and_anchor.json` |
| Appendix F full-cohort stay length: 0.4479; strata 0.7397 and 0.7522; recoded 0.6490 to 0.7691; shortest quintile 17.86 vs 11.06 | `horizon_and_anchor.json`, `stay_length_controls.json`, `anchor_and_stratification.json` (`q1`) |
| Table H1 ICD by cohort, deaths 3,444 and 4,211, mortality 8.73% and 10.12% | `revision/extended_checks.json` (`icd_version`), `icd.json` |
| Appendix I storage-time rows: 42,059 of 1,753,264 labs; 312,045 of 13,774,471 vitals | `task4_task5b.json` |
