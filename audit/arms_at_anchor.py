"""Table 1's arms, recomputed on the alive-at-24h cohort.

The published cohort applies a 24-hour feature window but does not require
survival to it, so for roughly one decedent in five the label is realised before
the prediction anchor. Section 4 discloses this and repairs arm (d) — the count
alone — but Table 1's multi-feature arms are still computed on the full cohort.
That matters in both directions: first-24h vitals and laboratory values for a
patient dying at hour 10 are agonal measurements, so arm (a) is contaminated
too, and the sign of the effect on the arm (a) -> (b) delta is not predictable
a priori.

This script rebuilds arms (a), (b) and (c) on admissions alive and still
inpatient at 24 hours (n = 81,052 under the predicate used in
`horizon_and_anchor.py`) and reports the same statistics Table 1 carries, so the
delta can be quoted at a valid anchor.

NOT RUNNABLE on a small machine: `build_cohort_frame` reads `icu/chartevents`
(54M rows) and `hosp/labevents` (37M rows). Budget roughly the same wall time as
`analysis.py`, about 40 minutes with 14 threads and 6 GB.

AGGREGATE OUTPUT ONLY. Out-of-fold predictions stay in memory; the output file
carries point estimates, intervals and counts.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    export DUCKDB_THREADS=14 DUCKDB_MEMORY=6GB
    python audit/arms_at_anchor.py
"""
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import arms as A  # noqa: E402
from cohort_build import build_cohort_frame  # noqa: E402

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
LABEL = "hospital_expire_flag"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")

    log("Building charttime cohort (full corpus)...")
    frame, meta = build_cohort_frame(M, con=con, time_col="charttime")

    # anchor predicate: alive AND still inpatient at 24 h after ICU admission.
    # Identical to horizon_and_anchor.py so the two cohorts agree.
    times = con.execute(f"""
        WITH f AS (SELECT hadm_id, intime,
                          ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
                   FROM read_csv_auto('{M}/icu/icustays.csv.gz'))
        SELECT f.hadm_id,
               date_diff('minute', f.intime, a.deathtime) / 1440.0 AS death_from_icu,
               date_diff('minute', f.intime, a.dischtime) / 1440.0 AS los_from_icu
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
        WHERE f.rn = 1""").df()
    frame = frame.merge(times, on="hadm_id", how="left")
    d_icu = np.where(frame.death_from_icu.notna().values,
                     frame.death_from_icu.values, np.inf)
    keep = ~((d_icu <= 1.0) | (frame.los_from_icu.values <= 1.0))
    frame = frame.loc[keep].reset_index(drop=True)

    y = frame[LABEL].to_numpy(int)
    groups = frame["subject_id"].to_numpy()
    log(f"  anchored cohort n={len(frame)} deaths={int(y.sum())} "
        f"subjects={len(np.unique(groups))}  (published cohort n={meta['n']})")

    out = {"corpus": "MIMIC-IV v3.1 credentialed",
           "anchor": "alive and inpatient at 24 h after ICU admission",
           "seed": A.SEED, "folds": A.FOLDS, "n_boot": A.N_BOOT,
           "n": int(len(frame)), "n_pos": int(y.sum()),
           "n_subjects": int(len(np.unique(groups)))}

    for arm in ("a", "b", "c"):
        cat_cols, num_cols = A._cols_for_arm(arm)
        log(f"arm ({arm})...")
        p = A._oof(frame, cat_cols, num_cols, y, groups)
        auroc = A._boot(y, p, groups, roc_auc_score)
        auprc = A._boot(y, p, groups, average_precision_score)
        slope, icpt = A._calib_si(y, p)
        out[arm] = {"auroc": {"point": round(auroc[0], 4), "lo95": round(auroc[1], 4),
                              "hi95": round(auroc[2], 4)},
                    "auprc": {"point": round(auprc[0], 4), "lo95": round(auprc[1], 4),
                              "hi95": round(auprc[2], 4)},
                    "brier": round(float(brier_score_loss(y, p)), 4),
                    "cal_slope": round(slope, 3), "cal_intercept": round(icpt, 3)}
        log(f"  arm ({arm}) AUROC={out[arm]['auroc']['point']:.4f} "
            f"[{out[arm]['auroc']['lo95']:.4f}, {out[arm]['auroc']['hi95']:.4f}]")

    out["delta_b_minus_a_auroc"] = round(
        out["b"]["auroc"]["point"] - out["a"]["auroc"]["point"], 4)
    out["delta_c_minus_a_auroc"] = round(
        out["c"]["auroc"]["point"] - out["a"]["auroc"]["point"], 4)
    log(f"  delta (a)->(b) = {out['delta_b_minus_a_auroc']:+.4f}   "
        f"(published cohort: +0.0129)")

    out["wall_seconds"] = round(time.time() - t0, 1)
    (OUT / "arms_at_anchor.json").write_text(json.dumps(out, indent=2))
    log("DONE -> outputs/arms_at_anchor.json")


if __name__ == "__main__":
    main()
