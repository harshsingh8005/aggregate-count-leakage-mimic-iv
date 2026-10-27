"""Full-corpus leakage-audit analysis (Tasks 2, 3, 5). AGGREGATE-ONLY output.

One in-memory process: builds the loader-equivalent W1 cohort on the
credentialed corpus, runs the three Task-2 arms (subject-grouped CV + cluster
bootstrap CIs, parameterized encoder verified == audited build_features), the
Task-3 n_diagnoses isolation test, and the Task-5 charttime-vs-storetime
analysis. Writes ONLY aggregate statistics to outputs/results.json.
NEVER writes row-level data, ids, or ICD codes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproducer_src"))
from cohort_build import build_cohort_frame  # noqa: E402
from src.model import fit_predict_proba  # noqa: E402
from src.loader import LABEL_COL, CATEGORICAL_COLS, LAB_ITEMIDS, VITAL_ITEMIDS  # noqa: E402
import arms as A  # noqa: E402  (verified parameterized encoder + CV + bootstrap)

ROOT = A.ROOT
SEED = A.SEED
FOLDS = A.FOLDS


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def task2(frame, y, groups, results):
    log("TASK 2: three-arm subject-grouped CV (parameterized encoder)")
    out = {}
    for arm in ("a", "b", "c"):
        cat_cols, num_cols = A._cols_for_arm(arm)
        oof = A._oof(frame, cat_cols, num_cols, y, groups)
        au = A._boot(y, oof, groups, lambda a, b: roc_auc_score(a, b))
        ap = A._boot(y, oof, groups, lambda a, b: average_precision_score(a, b))
        br = float(brier_score_loss(y, oof)); sl, ic = A._calib_si(y, oof)
        out[arm] = {"numeric_features": num_cols,
                    "auroc": {"point": au[0], "lo95": au[1], "hi95": au[2]},
                    "auprc": {"point": ap[0], "lo95": ap[1], "hi95": ap[2]},
                    "brier": br, "calib_slope": sl, "calib_intercept": ic}
        log(f"  arm {arm}: AUROC={au[0]:.4f} [{au[1]:.4f},{au[2]:.4f}] "
            f"AUPRC={ap[0]:.4f} Brier={br:.4f} slope={sl:.3f} icpt={ic:.3f}")
    results["task2"] = out


def task3(frame, y, groups, results):
    log("TASK 3: n_diagnoses isolation test (sole feature)")
    sub = frame[["n_diagnoses"]].astype(float)
    oof = np.full(len(y), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in sgkf.split(sub, y, groups):
        oof[te_idx] = fit_predict_proba(sub.iloc[tr_idx], y[tr_idx], sub.iloc[te_idx], random_state=SEED)
    au = A._boot(y, oof, groups, lambda a, b: roc_auc_score(a, b))
    nd = frame["n_diagnoses"].to_numpy()
    def q(mask):
        v = nd[mask]
        return {"n": int(mask.sum()), "mean": float(v.mean()), "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                "q95": float(np.percentile(v, 95)), "max": float(v.max())}
    los = frame["_los_days"].to_numpy()
    m = ~np.isnan(los)
    rho, pval = spearmanr(nd[m], los[m])
    results["task3"] = {"auroc_sole_feature": {"point": au[0], "lo95": au[1], "hi95": au[2]},
                        "distribution_by_class": {"survivors": q(y == 0), "deaths": q(y == 1)},
                        "spearman_ndx_vs_los": {"rho": float(rho), "p": float(pval),
                                                "los_source": "admissions.dischtime - admittime (days)"}}
    log(f"  n_diagnoses-only AUROC={au[0]:.4f} [{au[1]:.4f},{au[2]:.4f}] spearman={rho:.4f}")


def task5_gaps(con, results):
    log("TASK 5-critical: charttime vs storetime gap quantiles")
    out = {"pipeline_filters_on": "charttime (loader.py:287-289 vitals, 311-313 labs)"}
    for tbl, path, itemids in [
        ("labevents", str(Path(ROOT) / "hosp" / "labevents.csv.gz"),
         [i for ids in LAB_ITEMIDS.values() for i in ids]),
        ("chartevents", str(Path(ROOT) / "icu" / "chartevents.csv.gz"),
         [i for ids in VITAL_ITEMIDS.values() for i in ids]),
    ]:
        idlist = ",".join(str(i) for i in itemids)
        q = f"""
        SELECT quantile_cont(gap,0.25) q25, quantile_cont(gap,0.50) q50,
               quantile_cont(gap,0.75) q75, quantile_cont(gap,0.95) q95,
               quantile_cont(gap,0.99) q99, avg(gap) mean, count(*) n,
               sum(CASE WHEN gap<0 THEN 1 ELSE 0 END) n_negative
        FROM (SELECT date_diff('minute', charttime, storetime) AS gap
              FROM read_csv_auto('{path}', types={{'charttime':'TIMESTAMP','storetime':'TIMESTAMP'}})
              WHERE itemid IN ({idlist}) AND charttime IS NOT NULL AND storetime IS NOT NULL)
        """
        row = con.execute(q).df().iloc[0].to_dict()
        out[tbl + "_gap_minutes"] = {k: (float(v) if v is not None else None) for k, v in row.items()}
        log(f"  {tbl}: median={row['q50']:.1f} q95={row['q95']:.1f} q99={row['q99']:.1f} "
            f"n={int(row['n'])} n_neg={int(row['n_negative'])}")
    results["task5"] = out


import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

def main():
    t0 = time.time()
    results = {"corpus": "MIMIC-IV v3.1 credentialed", "seed": SEED, "folds": FOLDS,
               "n_boot": A.N_BOOT,
               "split": "subject_id (StratifiedGroupKFold); cluster bootstrap by subject_id"}
    con = duckdb.connect(); con.execute("PRAGMA threads=" + _os.environ.get("DUCKDB_THREADS","4")) ; con.execute("SET memory_limit='" + _os.environ.get("DUCKDB_MEMORY","6GB") + "'")

    log("Building charttime cohort (full corpus)...")
    frame, meta = build_cohort_frame(ROOT, con=con, time_col="charttime")
    results["cohort"] = meta
    log(f"  cohort n={meta['n']} n_pos={meta['n_pos']} subjects={meta['n_subjects']} "
        f"prevalence={meta['n_pos']/meta['n']:.4f}")

    los = con.execute(
        f"SELECT hadm_id, date_diff('second', admittime, dischtime)/86400.0 AS los "
        f"FROM read_csv_auto('{Path(ROOT)/'hosp'/'admissions.csv.gz'}', "
        f"types={{'admittime':'TIMESTAMP','dischtime':'TIMESTAMP'}})").df()
    frame = frame.merge(los.rename(columns={"los": "_los_days"}), on="hadm_id", how="left")

    y = frame[LABEL_COL].to_numpy(int); groups = frame["subject_id"].to_numpy()

    task2(frame, y, groups, results)
    task3(frame, y, groups, results)
    task5_gaps(con, results)

    # Task 5 continued: n_diagnoses knowability + storetime-windowed arm (a).
    log("Building storetime cohort (full corpus) for arm-(a) delta...")
    frame_s, meta_s = build_cohort_frame(ROOT, con=con, time_col="storetime")
    ys = frame_s[LABEL_COL].to_numpy(int); gs = frame_s["subject_id"].to_numpy()
    cat_cols, num_cols = A._cols_for_arm("a")
    oof_s = A._oof(frame_s, cat_cols, num_cols, ys, gs)
    au_s = A._boot(ys, oof_s, gs, lambda a, b: roc_auc_score(a, b))
    results["task5"]["arm_a_storetime"] = {
        "auroc": {"point": au_s[0], "lo95": au_s[1], "hi95": au_s[2]}, "cohort": meta_s,
        "auroc_delta_vs_charttime": au_s[0] - results["task2"]["a"]["auroc"]["point"]}
    log(f"  arm(a) storetime AUROC={au_s[0]:.4f} "
        f"delta={results['task5']['arm_a_storetime']['auroc_delta_vs_charttime']:+.4f}")

    results["wall_seconds"] = round(time.time() - t0, 1)
    Path(_OUT, "results.json").write_text(json.dumps(results, indent=2))
    log(f"DONE in {results['wall_seconds']}s -> outputs/results.json")


if __name__ == "__main__":
    main()
