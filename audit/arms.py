"""Task-2 three-arm re-run with a PARAMETERIZED encoder.

The audited features.py hardcodes its numeric column list (anchor_age,
charlson_history, vitals, labs) and never includes n_diagnoses, so it cannot
express the counterfactual arms. This module reproduces features.py's encoding
logic exactly (one-hot drop-first with a train vocabulary; per-column train
median imputation) but takes an explicit numeric column list, so the three arms
differ ONLY in their feature set. `verify_encoder_matches` proves it reproduces
the audited build_features on arm (a)'s column set.

AGGREGATE-ONLY output. No row-level data, ids, or ICD codes persisted.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproducer_src"))
from cohort_build import build_cohort_frame  # noqa: E402
from src.model import fit_predict_proba  # noqa: E402
from src.calibration import BetaBayesCalibrator  # noqa: E402
from src.loader import LABEL_COL, CATEGORICAL_COLS  # noqa: E402
from src.features import build_features  # noqa: E402 (for verification only)

ROOT = os.environ.get("MIMIC_IV_PATH", "")  # path to mimiciv/3.1; set before running
SEED = 42; FOLDS = 5; CAL_FRAC = 0.25; N_BOOT = 1000
VITALS = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "spo2", "temperature"]
LABS = ["creatinine", "wbc", "hemoglobin", "platelet", "glucose", "sodium",
        "potassium", "bicarbonate", "bun", "lactate"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _fit_encoder(frame, cat_cols, num_cols):
    cats = {}
    for c in cat_cols:
        vals = frame[c].astype("object").where(frame[c].notna(), other="__missing__")
        cats[c] = sorted(str(v) for v in vals.unique())
    meds = {c: (float(frame[c].astype(float).median()) if frame[c].notna().any() else 0.0)
            for c in num_cols}
    return cats, meds


def _transform(frame, cats, meds, num_cols):
    num = pd.DataFrame(index=frame.index)
    for c in num_cols:
        s = frame[c].astype(float) if c in frame.columns else pd.Series(np.nan, index=frame.index)
        num[c] = s.fillna(meds[c])
    blocks = []
    for c, levels in cats.items():
        vals = frame[c].astype("object").where(frame[c].notna(), other="__missing__")
        for lvl in levels[1:]:
            blocks.append((vals == lvl).astype(np.float64).rename(f"{c}={lvl}"))
    cat = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=frame.index)
    return pd.concat([num, cat], axis=1).astype(np.float64)


def _cols_for_arm(arm):
    if arm == "a":   num = ["anchor_age", "charlson_history"] + VITALS + LABS
    elif arm == "b": num = ["anchor_age", "charlson_history", "n_diagnoses"] + VITALS + LABS
    elif arm == "c": num = ["anchor_age", "n_diagnoses"] + VITALS + LABS
    return list(CATEGORICAL_COLS), num


def verify_encoder_matches(demo_root):
    """Prove the parameterized encoder == audited build_features on arm (a) cols."""
    f, _ = build_cohort_frame(demo_root)
    _, num_a = _cols_for_arm("a")
    cats, meds = _fit_encoder(f, list(CATEGORICAL_COLS), num_a)
    mine = _transform(f, cats, meds, num_a)
    ref = build_features(f)  # audited encoder (fit-transform on same frame)
    ref = ref.reindex(sorted(ref.columns), axis=1)
    mine = mine.reindex(sorted(mine.columns), axis=1)
    same_cols = list(ref.columns) == list(mine.columns)
    same_vals = same_cols and np.allclose(ref.values, mine[ref.columns].values, atol=1e-9)
    return {"same_cols": bool(same_cols), "same_vals": bool(same_vals),
            "ref_shape": ref.shape, "mine_shape": mine.shape}


def _oof(frame, cat_cols, num_cols, y, groups):
    oof = np.full(len(y), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in sgkf.split(frame, y, groups):
        tr = frame.iloc[tr_idx]; g_tr = groups[tr_idx]; y_tr = y[tr_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=SEED)
        fit_pos, cal_pos = next(gss.split(tr, y_tr, g_tr))
        cats, meds = _fit_encoder(tr.iloc[fit_pos], cat_cols, num_cols)
        Xf = _transform(tr.iloc[fit_pos], cats, meds, num_cols)
        Xc = _transform(tr.iloc[cal_pos], cats, meds, num_cols).reindex(columns=Xf.columns, fill_value=0.0)
        Xt = _transform(frame.iloc[te_idx], cats, meds, num_cols).reindex(columns=Xf.columns, fill_value=0.0)
        cs = fit_predict_proba(Xf, y_tr[fit_pos], Xc, random_state=SEED)
        ts = fit_predict_proba(Xf, y_tr[fit_pos], Xt, random_state=SEED)
        calib = BetaBayesCalibrator().fit(cs, y_tr[cal_pos])
        oof[te_idx] = calib.predict(ts)
    return oof


def _calib_si(y, p):
    eps = 1e-6; pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(lp, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _boot(y, p, groups, fn, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    by_g = {g: np.array(ix) for g, ix in by_g.items()}
    stats = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in samp])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2: continue
        stats.append(fn(yy, pp))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(fn(y, p)), float(lo), float(hi)


import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

def main():
    t0 = time.time()
    con = duckdb.connect(); con.execute("PRAGMA threads=" + os.environ.get("DUCKDB_THREADS","4"))
    log("Building charttime cohort...")
    frame, meta = build_cohort_frame(ROOT, con=con, time_col="charttime")
    log(f"  cohort n={meta['n']} n_pos={meta['n_pos']} subjects={meta['n_subjects']}")
    y = frame[LABEL_COL].to_numpy(int); groups = frame["subject_id"].to_numpy()
    out = {"cohort": meta, "arms": {}}
    for arm in ("a", "b", "c"):
        cat_cols, num_cols = _cols_for_arm(arm)
        oof = _oof(frame, cat_cols, num_cols, y, groups)
        au = _boot(y, oof, groups, lambda a, b: roc_auc_score(a, b))
        ap = _boot(y, oof, groups, lambda a, b: average_precision_score(a, b))
        br = float(brier_score_loss(y, oof)); sl, ic = _calib_si(y, oof)
        out["arms"][arm] = {"numeric_features": num_cols,
                            "auroc": {"point": au[0], "lo95": au[1], "hi95": au[2]},
                            "auprc": {"point": ap[0], "lo95": ap[1], "hi95": ap[2]},
                            "brier": br, "calib_slope": sl, "calib_intercept": ic}
        log(f"  arm {arm}: AUROC={au[0]:.4f} [{au[1]:.4f},{au[2]:.4f}] AUPRC={ap[0]:.4f} Brier={br:.4f} slope={sl:.3f} icpt={ic:.3f}")
    out["wall_seconds"] = round(time.time() - t0, 1)
    Path(_OUT, "arms_results.json").write_text(json.dumps(out, indent=2))
    log(f"DONE {out['wall_seconds']}s -> arms_results.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        demo = Path(os.environ.get("MIMIC_IV_DEMO_PATH", ""))
        print("ENCODER_VERIFY", verify_encoder_matches(demo))
    else:
        main()
