"""Paired subject-cluster bootstrap on the arm (a) -> (b) and (a) -> (c) deltas.

The manuscript's headline comparison is a difference between two models fitted
on the same admissions. Reporting two marginal confidence intervals and noting
that they do not overlap is a conservative test of that difference, not a
direct one: it ignores the positive correlation between the two arms' errors
and so understates the evidence. This script computes the difference directly.

Both arms are scored out-of-fold under the identical StratifiedGroupKFold
split, so every admission has a prediction from each arm. Each bootstrap
replicate resamples *subjects* with replacement and recomputes both arms'
AUROC and AUPRC on that same resample, giving a distribution of the paired
difference. The reported interval is the 2.5/97.5 percentile of that
distribution, and the reported p-value is the two-sided proportion of
replicates in which the difference reverses sign.

AGGREGATE OUTPUT ONLY. Out-of-fold predictions are held in memory and never
written; the output file contains point estimates, intervals and counts.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/paired_delta.py
"""
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import arms as A  # noqa: E402
from cohort_build import build_cohort_frame  # noqa: E402

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")

LABEL = "hospital_expire_flag"
N_BOOT = A.N_BOOT
SEED = A.SEED


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(y, pa, pb, groups, fn, n=N_BOOT, seed=SEED):
    """Distribution of fn(b) - fn(a) under a shared subject-cluster resample."""
    rng = np.random.default_rng(seed)
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    by_g = {g: np.asarray(ix) for g, ix in by_g.items()}
    uniq = np.fromiter(by_g.keys(), dtype=np.asarray(groups).dtype)

    deltas = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in samp])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(fn(yy, pb[idx]) - fn(yy, pa[idx]))
    deltas = np.asarray(deltas)
    point = fn(y, pb) - fn(y, pa)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p: proportion of replicates on the far side of zero
    frac = float(np.mean(deltas <= 0)) if point > 0 else float(np.mean(deltas >= 0))
    p = min(1.0, 2 * frac)
    return {
        "point": round(float(point), 5),
        "lo95": round(float(lo), 5),
        "hi95": round(float(hi), 5),
        "p_two_sided": round(p, 5),
        "n_replicates": int(len(deltas)),
        "frac_reversing_sign": round(frac, 5),
    }


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=" + os.environ.get("DUCKDB_THREADS","4"))

    log("Building charttime cohort (full corpus)...")
    frame, meta = build_cohort_frame(M, con=con, time_col="charttime")
    y = frame[LABEL].to_numpy(int)
    groups = frame["subject_id"].to_numpy()
    log(f"  cohort n={meta['n']} n_pos={meta['n_pos']} subjects={meta['n_subjects']}")

    oof = {}
    for arm in ("a", "b", "c"):
        cat_cols, num_cols = A._cols_for_arm(arm)
        log(f"Out-of-fold predictions, arm ({arm})...")
        oof[arm] = A._oof(frame, cat_cols, num_cols, y, groups)
        log(f"  arm ({arm}) AUROC={roc_auc_score(y, oof[arm]):.4f}")

    out = {
        "corpus": "MIMIC-IV v3.1 credentialed",
        "seed": SEED,
        "n_boot": N_BOOT,
        "design": ("identical StratifiedGroupKFold split across arms; each "
                   "bootstrap replicate resamples subjects once and scores "
                   "both arms on that same resample"),
        "cohort": meta,
    }
    for lo_arm, hi_arm in (("a", "b"), ("a", "c")):
        for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            key = f"delta_{hi_arm}_minus_{lo_arm}_{name}"
            out[key] = paired_boot(y, oof[lo_arm], oof[hi_arm], groups, fn)
            r = out[key]
            log(f"  {key}: {r['point']:+.5f} [{r['lo95']:+.5f}, {r['hi95']:+.5f}] "
                f"p={r['p_two_sided']:.4f}")

    out["wall_seconds"] = round(time.time() - t0, 1)
    dest = HERE.parent / "outputs" / "paired_delta.json"
    dest.write_text(json.dumps(out, indent=2))
    log(f"DONE in {out['wall_seconds']}s -> {dest}")


if __name__ == "__main__":
    main()
