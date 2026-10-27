"""Classifier and split robustness of the +0.017 gain, for "A Count Is Not a Code".

Does the gain from adding the count survive a different classifier, per-set
tuning, and splits that approximate calendar time?

On the 24-hour cohort (alive and in hospital 24 h after ICU admission, n = 81,052,
the predicate of arms_at_anchor.py) this script compares feature set (a), the
cleaned set, with (b) = (a) + the distinct-code count, under:

  M0  the submitted pipeline (fixed LightGBM settings + Beta calibration).
      REPRODUCTION GATE: must give (a) 0.8388 and (b) 0.8558 (arms_at_anchor.json).
  M1  L2 logistic regression, standardised inputs, C chosen by inner
      subject-grouped CV from {0.01, 0.1, 1, 10}.
  M2  LightGBM tuned SEPARATELY for each feature set by inner subject-grouped CV
      over a 12-point grid, so neither set is favoured by settings chosen for
      the other. This is a robustness check: the paper (Appendix E) holds settings
      fixed on purpose, because per-set tuning mixes a feature effect with a
      settings effect. Read M0 and M2 together.
  M4  the split "like Ramadan et al.": latest 20% of admissions by (shifted) admission
      date as test, overlapping subjects dropped from training. Not chronological in
      MIMIC-IV, because dates are shifted per patient; reported for direct comparability.
  M3  approximate time-based split: train on patients whose anchor_year_group is
      2016 or earlier, test on later groups (MIMIC-IV shifts dates per patient,
      so this is the closest available to a calendar split). Submitted settings.

M0-M2 share the outer split of the submitted pipeline (StratifiedGroupKFold,
5 folds, seed 42). The gain (b) - (a) is reported with a paired subject-cluster
bootstrap (1000 replicates), as paired_delta.py does. AUROC is invariant to the
monotone calibration step, so M1-M3 skip it.

NOT A SMALL-MACHINE JOB: building the frame reads icu/chartevents (54M rows)
and hosp/labevents. Budget about 40 min for the frame plus about 30 min for M2.

AGGREGATE OUTPUT ONLY. Out-of-fold predictions stay in memory.

Usage (same environment as the supplementary scripts):
    set MIMIC_IV_PATH=...\\physionet.org\\files\\mimiciv\\3.1
    set DUCKDB_THREADS=14 & set DUCKDB_MEMORY=6GB
    python revision/robustness_checks.py
Writes outputs/revision/robustness_checks.json under the repository root.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = ROOT / "audit"
sys.path.insert(0, str(AUDIT))
import arms as A  # noqa: E402  (puts reproducer_src on sys.path itself)

OUT = ROOT / "outputs" / "revision"
LABEL = "hospital_expire_flag"
SEED, FOLDS, N_BOOT = A.SEED, A.FOLDS, 1000
GATE = {"a": 0.8388, "b": 0.8558, "n": 81052, "deaths": 7659}
TOL = 0.0006
LGB_GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
            for nl, lr, ne, mc in itertools.product((15, 63), (0.03, 0.1), (300,), (20, 100, 400))]
LR_GRID = (0.01, 0.1, 1.0, 10.0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(y, pa, pb, groups, n=N_BOOT, seed=SEED):
    """As paired_delta.py: shared subject-cluster resample, delta = AUROC(b) - AUROC(a)."""
    rng = np.random.default_rng(seed)
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    idx_by = [np.asarray(v) for v in by_g.values()]
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[j] for j in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx]))
    d = np.asarray(d)
    point = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    lo, hi = np.percentile(d, [2.5, 97.5])
    frac = float(np.mean(d <= 0)) if point > 0 else float(np.mean(d >= 0))
    return {"auroc_a": round(float(roc_auc_score(y, pa)), 4),
            "auroc_b": round(float(roc_auc_score(y, pb)), 4),
            "delta": round(float(point), 5), "lo95": round(float(lo), 5),
            "hi95": round(float(hi), 5), "frac_reversing_sign": round(frac, 5),
            "n_replicates": int(len(d))}


def encode(train, test, cat_cols, num_cols):
    cats, meds = A._fit_encoder(train, cat_cols, num_cols)
    Xtr = A._transform(train, cats, meds, num_cols)
    Xte = A._transform(test, cats, meds, num_cols).reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr, Xte


def fit_lr(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def fit_lgb(Xtr, ytr, Xte, params):
    from lightgbm import LGBMClassifier
    p = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             n_jobs=int(os.environ.get("LGB_THREADS", "-1")), verbose=-1, random_state=SEED)
    p.update(params)
    return LGBMClassifier(**p).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def inner_select(frame, y, groups, cat_cols, num_cols, grid, fitter):
    """Pick the grid point with the best mean inner-CV AUROC (3 subject-grouped folds)."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(inner.split(frame, y, groups))
    best, best_auc = None, -1.0
    for g in grid:
        aucs = []
        for tr, va in splits:
            Xtr, Xva = encode(frame.iloc[tr], frame.iloc[va], cat_cols, num_cols)
            aucs.append(roc_auc_score(y[va], fitter(Xtr, y[tr], Xva, g)))
        if np.mean(aucs) > best_auc:
            best, best_auc = g, float(np.mean(aucs))
    return best, best_auc


def oof_tuned(frame, y, groups, arm, kind):
    cat_cols, num_cols = A._cols_for_arm(arm)
    grid, fitter = (LR_GRID, fit_lr) if kind == "lr" else (LGB_GRID, fit_lgb)
    oof = np.full(len(y), np.nan)
    chosen = []
    outer = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(outer.split(frame, y, groups)):
        ftr = frame.iloc[tr].reset_index(drop=True)
        best, inner_auc = inner_select(ftr, y[tr], groups[tr], cat_cols, num_cols, grid, fitter)
        Xtr, Xte = encode(ftr, frame.iloc[te], cat_cols, num_cols)
        oof[te] = fitter(Xtr, y[tr], Xte, best)
        chosen.append({"fold": k, "params": best, "inner_auroc": round(inner_auc, 4)})
        log(f"    {kind} arm ({arm}) fold {k}: {best} inner {inner_auc:.4f}")
    return oof, chosen


def anchor_frame(M, synthetic=0):
    if synthetic:
        rng = np.random.default_rng(0)
        n = synthetic
        cat_cols, num_cols = A._cols_for_arm("b")
        f = pd.DataFrame({c: rng.choice(["x", "y", "z", None], n) for c in cat_cols})
        for c in num_cols:
            f[c] = rng.normal(size=n)
        f["subject_id"] = rng.integers(0, n // 2, n)
        f["hadm_id"] = np.arange(n)
        f[LABEL] = (rng.random(n) < 1 / (1 + np.exp(-(f.n_diagnoses + f.bun - 2)))).astype(int)
        f["adm_time"] = pd.Timestamp("2150-01-01") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="m")
        f["anchor_year_group"] = rng.choice(
            ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"], n)
        return f, {"synthetic": n}
    import duckdb
    from cohort_build import build_cohort_frame
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")
    log("Building charttime frame (full corpus; reads chartevents and labevents)...")
    frame, meta = build_cohort_frame(M, con=con, time_col="charttime")
    times = con.execute(f"""
        WITH f AS (SELECT hadm_id, intime,
                          ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
                   FROM read_csv_auto('{M}/icu/icustays.csv.gz'))
        SELECT f.hadm_id, a.admittime AS adm_time,
               date_diff('minute', f.intime, a.deathtime) / 1440.0 AS death_from_icu,
               date_diff('minute', f.intime, a.dischtime) / 1440.0 AS los_from_icu
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
        WHERE f.rn = 1""").df()
    frame = frame.merge(times, on="hadm_id", how="left")
    d_icu = np.where(frame.death_from_icu.notna().values, frame.death_from_icu.values, np.inf)
    keep = ~((d_icu <= 1.0) | (frame.los_from_icu.values <= 1.0))
    frame = frame.loc[keep].reset_index(drop=True)
    ayg = con.execute(f"SELECT subject_id, anchor_year_group FROM "
                      f"read_csv_auto('{M}/hosp/patients.csv.gz')").df()
    frame = frame.merge(ayg, on="subject_id", how="left")
    return frame, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="test run on fake data")
    ap.add_argument("--skip-tuned-lgb", action="store_true")
    a = ap.parse_args()
    M = os.environ.get("MIMIC_IV_PATH", "")
    if not M and not a.synthetic:
        raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root (…/mimiciv/3.1)")
    M = M.replace("\\", "/").rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frame, meta = anchor_frame(M, a.synthetic)
    y = frame[LABEL].to_numpy(int)
    groups = frame["subject_id"].to_numpy()
    out = {"cohort": "alive and in hospital 24 h after ICU admission",
           "n": int(len(y)), "deaths": int(y.sum()), "seed": SEED, "n_boot": N_BOOT,
           "synthetic": bool(a.synthetic), "gates": []}
    log(f"24-h cohort n={len(y)} deaths={int(y.sum())}")

    def gate(name, got, want):
        ok = (abs(got - want) <= TOL) if isinstance(want, float) else got == want
        out["gates"].append({"gate": name, "submitted": want, "recomputed": got, "pass": bool(ok)})
        log(f"  GATE {'PASS' if ok else 'FAIL'}  {name}: submitted {want} recomputed {got}")

    gate("24-h cohort n", int(len(y)), GATE["n"])
    gate("24-h cohort deaths", int(y.sum()), GATE["deaths"])

    # M0: submitted pipeline, reproduction gate
    log("M0 submitted pipeline")
    oof0 = {arm: A._oof(frame, *A._cols_for_arm(arm), y, groups) for arm in ("a", "b")}
    gate("M0 (a) AUROC", round(float(roc_auc_score(y, oof0["a"])), 4), GATE["a"])
    gate("M0 (b) AUROC", round(float(roc_auc_score(y, oof0["b"])), 4), GATE["b"])
    out["M0_submitted_settings"] = paired_boot(y, oof0["a"], oof0["b"], groups)
    log(f"  M0 {out['M0_submitted_settings']}")

    # M1: logistic regression
    log("M1 L2 logistic regression, C by inner CV")
    lr = {arm: oof_tuned(frame, y, groups, arm, "lr") for arm in ("a", "b")}
    out["M1_logistic_regression"] = paired_boot(y, lr["a"][0], lr["b"][0], groups)
    out["M1_logistic_regression"]["chosen"] = {arm: lr[arm][1] for arm in lr}
    log(f"  M1 {out['M1_logistic_regression']}")

    # M2: LightGBM tuned per feature set
    if not a.skip_tuned_lgb:
        log("M2 LightGBM tuned per feature set (inner CV, 12-point grid)")
        lg = {arm: oof_tuned(frame, y, groups, arm, "lgb") for arm in ("a", "b")}
        out["M2_lightgbm_tuned_per_set"] = paired_boot(y, lg["a"][0], lg["b"][0], groups)
        out["M2_lightgbm_tuned_per_set"]["chosen"] = {arm: lg[arm][1] for arm in lg}
        log(f"  M2 {out['M2_lightgbm_tuned_per_set']}")

    # M3: approximate time-based split on anchor_year_group
    log("M3 time-based split by anchor_year_group (submitted settings)")
    ayg = frame["anchor_year_group"].astype(str)
    start = pd.to_numeric(ayg.str.slice(0, 4), errors="coerce")
    tr = (start <= 2014).to_numpy()
    te = (start >= 2017).to_numpy()
    from src.model import fit_predict_proba
    pt = {}
    for arm in ("a", "b"):
        cat_cols, num_cols = A._cols_for_arm(arm)
        Xtr, Xte = encode(frame.loc[tr], frame.loc[te], cat_cols, num_cols)
        pt[arm] = fit_predict_proba(Xtr, y[tr], Xte, random_state=SEED)
    m3 = paired_boot(y[te], pt["a"], pt["b"], groups[te])
    m3.update({"train_groups": sorted(ayg[tr].unique().tolist()),
               "test_groups": sorted(ayg[te].unique().tolist()),
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "deaths_test": int(y[te].sum()),
               "note": "patients split by anchor_year_group; approximate, dates are shifted per patient"})
    out["M3_time_split"] = m3
    log(f"  M3 {m3}")

    # M4: Ramadan-style split on (shifted) admission date, latest 20% as test.
    # Subjects with any test admission are dropped from training. MIMIC-IV shifts each
    # patient's dates independently, so this is not chronological; reported because
    # the split "like Ramadan et al.".
    log("M4 split on shifted admission date (latest 20% test), submitted settings")
    at = pd.to_datetime(frame["adm_time"])
    cut = at.quantile(0.8)
    te4 = (at > cut).to_numpy()
    test_subj = set(groups[te4])
    tr4 = (~te4) & ~np.isin(groups, list(test_subj))
    pt4 = {}
    for arm in ("a", "b"):
        cat_cols, num_cols = A._cols_for_arm(arm)
        Xtr, Xte = encode(frame.loc[tr4], frame.loc[te4], cat_cols, num_cols)
        pt4[arm] = fit_predict_proba(Xtr, y[tr4], Xte, random_state=SEED)
    m4 = paired_boot(y[te4], pt4["a"], pt4["b"], groups[te4])
    m4.update({"n_train": int(tr4.sum()), "n_test": int(te4.sum()), "deaths_test": int(y[te4].sum()),
               "train_subjects_dropped_for_overlap": int(((~te4) & ~tr4).sum()),
               "note": "latest 20% of admissions by shifted admittime; not chronological in MIMIC-IV"})
    out["M4_admission_date_split"] = m4
    log(f"  M4 {m4}")

    failed = [g["gate"] for g in out["gates"] if not g["pass"]]
    out["gates_failed"] = failed
    out["wall_seconds"] = round(time.time() - t0, 1)
    dest = OUT / ("robustness_checks_SYNTHETIC.json" if a.synthetic else "robustness_checks.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    log(f"{len(out['gates']) - len(failed)}/{len(out['gates'])} gates passed"
        + (f"; FAILED: {failed}" if failed else ""))
    log(f"DONE in {out['wall_seconds']} s -> {dest}")


if __name__ == "__main__":
    main()
