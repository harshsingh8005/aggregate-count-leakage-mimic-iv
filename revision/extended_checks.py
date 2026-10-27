"""Extended checks for "A Count Is Not a Code" (SSRN preprint, 2026).

Analyses added after the first version of the paper.
Everything here needs only cheap MIMIC-IV tables (no chartevents, no labevents).

  E1  cohort flow: exact composition of the 4,190 admissions removed by
               the 24-hour predicate (the "1,767 ... plus 49" sentence).
      one time origin: fixed-horizon labels and stay length measured from
               ICU admission (t0), and the stay-length controls and Appendix C
               chronic-code analyses re-run on the 24-hour cohort.
      ICD-9 / ICD-10 split on the 24-hour cohort.
  E2  other admission-level counts (procedures, prescriptions, orders,
               laboratory tests, microbiology specimens, radiology notes), each built over the whole
               admission and over data timestamped no later than t0 + 24 h.

REPRODUCTION GATES. Before any new number is trusted, the script recomputes the
submitted values with the submitted definitions (same SQL, same fold assignment)
and prints PASS/FAIL against them. If a gate fails, do not quote the new numbers.

AGGREGATE OUTPUT ONLY. Counts, rates and AUROCs. No identifier, timestamp or
row-level value is printed or written. Everything stays on this machine.

Usage (same environment as the supplementary audit scripts):
    set MIMIC_IV_PATH=...\\physionet.org\\files\\mimiciv\\3.1
    set MIMIC_IV_NOTE_PATH=...\\physionet.org\\files\\mimic-iv-note\\2.2   (optional)
    python revision/extended_checks.py                 # E1 + E2
    python revision/extended_checks.py --e1-only       # minutes; small tables only
    python revision/extended_checks.py --skip-poe      # POE is the largest table used
Writes outputs/revision/extended_checks.json under the repository root.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHARLSON_PY = ROOT / "reproducer_src" / "src" / "charlson.py"
OUT = ROOT / "outputs" / "revision"
SEED = 42          # as stay_length_controls.py
N_BINS = 20        # as stay_length_controls.py
N_BOOT = 200       # subject-cluster bootstrap for the E2 counts
N_RAND = 20        # random code subsets for the Appendix C null (submitted paper: 3)
TOL = 0.0006       # gate tolerance on 4-decimal published values

# Submitted values (supplement outputs/*.json). Gates compare against these.
PUBLISHED = {
    "n": 85242, "deaths": 9475, "died_within_24h_icu": 1767,
    "anchor_n": 81052, "anchor_deaths": 7659,
    "auroc_count_full": 0.6941, "auroc_count_anchor": 0.7288,
    "h_adm_7d": (5166, 0.5681), "h_adm_14d": (7197, 0.6280), "h_adm_30d": (8839, 0.6755),
    "anchor_cond_count_los_5": 0.7389, "anchor_cond_count_los_500": 0.7421,
    "anchor_cond_los_los_5": 0.4870, "anchor_cond_los_los_500": 0.5029,
    "full_recoded_los": 0.6490, "full_oof_recoded_plus_count": 0.7691,
    "full_chronic_only": 0.6453, "full_nonchronic": 0.6794,
    "full_cond_chronic_given_total": 0.5489, "full_cond_total_given_chronic": 0.6464,
    "icd9": (41395, 0.6562), "icd10": (43823, 0.7290),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- statistics: copied from the supplement so the definitions are identical ---
def rank_auroc(score, y):
    """Mann-Whitney AUROC with mid-ranks for ties (horizon_and_anchor.py)."""
    s = np.asarray(score, float)
    y = np.asarray(y, int)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def conditional_c(score, strata, y):
    """Pair-weighted concordance within strata (horizon_and_anchor.py)."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        yy = y[m]
        n1 = yy.sum()
        n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        num += rank_auroc(score[m], yy) * n1 * n0
        den += n1 * n0
    return float(num / den)


def strata_of(x, k):
    e = np.quantile(x, np.linspace(0, 1, k + 1))
    e[0] -= 1
    e[-1] += 1
    return np.clip(np.digitize(x, e) - 1, 0, k - 1)


def subject_folds(groups, seed=SEED):
    """Subject-grouped 5-fold assignment exactly as stay_length_controls.py."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fmap = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    return np.array([fmap[s] for s in groups])


def recoded_and_oof(los, cnt, y, fold):
    """Cross-fitted 20-bin stay-length death rate; OOF logistic with/without count."""
    e = np.quantile(los, np.linspace(0, 1, N_BINS + 1))
    e[0] -= 1
    e[-1] += 1
    b = np.clip(np.digitize(los, e) - 1, 0, N_BINS - 1)
    rec = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        rate = {i: (y[tr & (b == i)].mean() if (tr & (b == i)).sum() else y[tr].mean())
                for i in range(N_BINS)}
        rec[te] = [rate[i] for i in b[te]]

    def oof(X):
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            p[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return rank_auroc(p, y)

    return {"auroc_recoded_los": round(rank_auroc(rec, y), 4),
            "cond_count_given_recoded_values": round(conditional_c(cnt, rec, y), 4),
            "oof_recoded_los": round(oof(rec.reshape(-1, 1)), 4),
            "oof_recoded_los_plus_count": round(oof(np.column_stack([rec, cnt])), 4)}


def boot_ci(score, y, groups, n=N_BOOT, seed=SEED):
    """Subject-cluster bootstrap 95% CI of the oriented AUROC."""
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    idx_by = [np.asarray(v) for v in by_g.values()]
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, score)
    sign = 1 if point >= 0.5 else -1
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[j] for j in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, score[idx])
        vals.append(a if sign > 0 else 1 - a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def gate(results, name, got, want, tol=TOL):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    results.append({"gate": name, "submitted": want, "recomputed": got, "pass": bool(ok)})
    log(f"  GATE {'PASS' if ok else 'FAIL'}  {name}: submitted {want}  recomputed {got}")
    return ok


def load_charlson():
    spec = importlib.util.spec_from_file_location("charlson_local", CHARLSON_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CHARLSON_ICD9, mod.CHARLSON_ICD10


def csv_src(path):
    """All-varchar reader for the larger tables: no type sniffing to go wrong."""
    return f"read_csv('{path}', header=true, all_varchar=true)"


# ---------------------------------------------------------------- E1 ---------
def run_e1(con, M, out):
    log("E1: cohort, flow, single time origin")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE c AS
    WITH f AS (SELECT hadm_id, subject_id, stay_id, intime,
                      ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime, stay_id) rn
               FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
         d AS (SELECT hadm_id, COUNT(DISTINCT icd_code) n_dx, MAX(icd_version) icdv
               FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
    SELECT f.hadm_id, f.subject_id, f.intime AS t0,
           f.intime + INTERVAL 24 HOUR AS t24,
           a.hospital_expire_flag AS y, COALESCE(d.n_dx, 0) AS n_dx, d.icdv,
           date_diff('minute', a.admittime, a.dischtime) / 1440.0 AS los_adm,
           date_diff('minute', f.intime,   a.dischtime) / 1440.0 AS los_t0,
           CASE WHEN a.deathtime IS NULL THEN NULL
                ELSE date_diff('minute', a.admittime, a.deathtime) / 1440.0 END AS death_from_adm,
           CASE WHEN a.deathtime IS NULL THEN NULL
                ELSE date_diff('minute', f.intime, a.deathtime) / 1440.0 END AS death_from_icu,
           CASE WHEN a.deathtime IS NULL THEN NULL
                ELSE date_diff('minute', a.deathtime, a.dischtime) END AS disch_minus_death_min
    FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
           LEFT JOIN d USING (hadm_id)
    WHERE f.rn = 1
    ORDER BY f.hadm_id""")
    df = con.execute("""SELECT hadm_id, subject_id, y, n_dx, icdv, los_adm, los_t0,
                               death_from_adm, death_from_icu, disch_minus_death_min
                        FROM c ORDER BY hadm_id""").df()
    y = df.y.values.astype(int)
    cnt = df.n_dx.values.astype(float)
    g = df.subject_id.values
    dfi = df.death_from_icu.values.astype(float)
    dfa = df.death_from_adm.values.astype(float)
    los_t0 = df.los_t0.values.astype(float)
    los_adm = df.los_adm.values.astype(float)
    dmd = df.disch_minus_death_min.values.astype(float)
    gates = []
    out["gates"] = gates

    gate(gates, "cohort n", int(len(df)), PUBLISHED["n"])
    gate(gates, "deaths", int(y.sum()), PUBLISHED["deaths"])

    # ---- flow. Predicate exactly as arms_at_anchor.py
    died24 = ~np.isnan(dfi) & (dfi <= 1.0)
    disch24 = los_t0 <= 1.0
    anchor = ~(died24 | disch24)
    dec_extra = (y == 1) & disch24 & ~died24
    flow = {
        "cohort_n": int(len(df)), "cohort_deaths": int(y.sum()),
        "cohort_survivors": int((y == 0).sum()),
        "died_within_24h_of_icu_admission": int(died24.sum()),
        "  of which hospital_expire_flag=1": int((died24 & (y == 1)).sum()),
        "  of which death time before ICU admission": int((~np.isnan(dfi) & (dfi < 0)).sum()),
        "decedents_discharged_within_24h_not_died_within_24h": int(dec_extra.sum()),
        "  of which no recorded death time": int((dec_extra & np.isnan(dfi)).sum()),
        "  of which death recorded after discharge": int(
            (dec_extra & ~np.isnan(dmd) & (dmd < 0)).sum()),
        "  of which other": int((dec_extra & ~np.isnan(dfi) & ~(dmd < 0)).sum()),
        "survivors_discharged_within_24h": int(((y == 0) & disch24 & ~died24).sum()),
        "survivors_with_a_death_time (expect 0)": int(((y == 0) & ~np.isnan(dfi)).sum()),
        "decedents_with_no_death_time": int(((y == 1) & np.isnan(dfi)).sum()),
        "anchor_n": int(anchor.sum()), "anchor_deaths": int(y[anchor].sum()),
        "anchor_survivors": int((y[anchor] == 0).sum()),
    }
    out["flow"] = flow
    for k, v in flow.items():
        log(f"  {k}: {v:,}")
    gate(gates, "died within 24 h of ICU admission", flow["died_within_24h_of_icu_admission"],
         PUBLISHED["died_within_24h_icu"])
    gate(gates, "24-h cohort n", flow["anchor_n"], PUBLISHED["anchor_n"])
    gate(gates, "24-h cohort deaths", flow["anchor_deaths"], PUBLISHED["anchor_deaths"])
    gate(gates, "count AUROC, submitted cohort", round(rank_auroc(cnt, y), 4),
         PUBLISHED["auroc_count_full"])
    ya, ca, ga = y[anchor], cnt[anchor], g[anchor]
    gate(gates, "count AUROC, 24-h cohort", round(rank_auroc(ca, ya), 4),
         PUBLISHED["auroc_count_anchor"])

    # ---- fixed horizons: submitted (from hospital admission) vs t0
    hz = {"from_hospital_admission_submitted_cohort": {}, "from_t0_24h_cohort": {},
          "from_t0_submitted_cohort": {}}
    for H in (7, 14, 30):
        yh = (~np.isnan(dfa) & (dfa <= H)).astype(int)
        a_ = round(rank_auroc(cnt, yh), 4)
        hz["from_hospital_admission_submitted_cohort"][f"{H}d"] = {"events": int(yh.sum()), "auroc_count": a_}
        ev, au = PUBLISHED[f"h_adm_{H}d"]
        gate(gates, f"horizon {H}d from hospital admission (events)", int(yh.sum()), ev)
        gate(gates, f"horizon {H}d from hospital admission (AUROC)", a_, au)
        yt = (~np.isnan(dfi) & (dfi <= H)).astype(int)
        hz["from_t0_submitted_cohort"][f"{H}d"] = {"events": int(yt.sum()),
                                                   "auroc_count": round(rank_auroc(cnt, yt), 4)}
        yta = yt[anchor]
        hz["from_t0_24h_cohort"][f"{H}d"] = {"events": int(yta.sum()),
                                             "auroc_count": round(rank_auroc(ca, yta), 4)}
    hz["from_t0_24h_cohort"]["in_hospital"] = {"events": int(ya.sum()),
                                               "auroc_count": round(rank_auroc(ca, ya), 4)}
    out["fixed_horizon"] = hz
    log(f"  horizons from t0, 24-h cohort: {hz['from_t0_24h_cohort']}")

    # ---- stay-length controls. Gate the submitted definitions first.
    sl = {}
    an = {"n": int(anchor.sum()), "deaths": int(ya.sum())}
    la_adm = los_adm[anchor]
    for k in (5, 500):
        v = round(conditional_c(ca, strata_of(la_adm, k), ya), 4)
        gate(gates, f"24-h cohort: count | hospital-LOS strata k={k}", v,
             PUBLISHED[f"anchor_cond_count_los_{k}"])
        v2 = round(conditional_c(la_adm, strata_of(la_adm, k), ya), 4)
        gate(gates, f"24-h cohort: LOS | LOS strata k={k} (null control)", v2,
             PUBLISHED[f"anchor_cond_los_los_{k}"])
    full_fold = subject_folds(g)
    full = recoded_and_oof(los_adm, cnt, y, full_fold)
    gate(gates, "submitted cohort: recoded LOS alone", full["auroc_recoded_los"],
         PUBLISHED["full_recoded_los"])
    gate(gates, "submitted cohort: OOF recoded LOS + count", full["oof_recoded_los_plus_count"],
         PUBLISHED["full_oof_recoded_plus_count"])
    sl["submitted_cohort_hospital_los (reproduction)"] = full

    # New: 24-h cohort, stay length measured from t0
    lt = los_t0[anchor]
    an["auroc_los_t0"] = round(rank_auroc(lt, ya), 4)
    an["auroc_count"] = round(rank_auroc(ca, ya), 4)
    an["spearman_count_los_t0"] = round(float(spearmanr(ca, lt).statistic), 4)
    for k in (1, 5, 20, 100, 500):
        an[f"cond_count_given_los_t0_{k}"] = round(conditional_c(ca, strata_of(lt, k), ya), 4)
        an[f"cond_los_t0_given_los_t0_{k}"] = round(conditional_c(lt, strata_of(lt, k), ya), 4)
    an.update(recoded_and_oof(lt, ca, ya, subject_folds(ga)))
    sl["24h_cohort_los_from_t0 (new)"] = an
    out["stay_length_controls"] = sl
    log(f"  24-h cohort, LOS from t0: {an}")

    # ---- Appendix C on both cohorts
    C9, C10 = load_charlson()
    p9 = tuple(sorted({p for v in C9.values() for p in v}))
    p10 = tuple(sorted({p for v in C10.values() for p in v}))
    dxr = con.execute(f"""SELECT d.hadm_id, d.icd_code, d.icd_version
        FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') d
        JOIN c USING (hadm_id) ORDER BY d.hadm_id, d.icd_version, d.icd_code""").df()
    chronic = np.array([str(cd).strip().startswith(p10 if int(v) == 10 else p9)
                        for cd, v in zip(dxr.icd_code, dxr.icd_version)])
    rate = float(chronic.mean())
    hadm = df.hadm_id

    def per_adm(mask):
        return hadm.map(dxr[mask].groupby("hadm_id").icd_code.nunique()).fillna(0).values.astype(float)

    chc, nonc = per_adm(chronic), per_adm(~chronic)
    rng2 = np.random.default_rng(0)
    rand = [per_adm(rng2.random(len(dxr)) < rate) for _ in range(N_RAND)]
    ac = {}
    for tag, m in (("submitted_cohort", np.ones(len(y), bool)), ("24h_cohort", anchor)):
        yy, cc = y[m], cnt[m]
        ac[tag] = {
            "auroc_chronic_only": round(rank_auroc(chc[m], yy), 4),
            "auroc_nonchronic": round(rank_auroc(nonc[m], yy), 4),
            "auroc_random_subset_same_rate": [round(rank_auroc(r[m], yy), 4) for r in rand],
            "random_subset_min_max_mean": None,
            "cond_chronic_given_total": round(conditional_c(chc[m], cc, yy), 4),
            "cond_total_given_chronic": round(conditional_c(cc, chc[m], yy), 4),
            "mean_chronic_codes_deaths": round(float(chc[m][yy == 1].mean()), 2),
            "mean_chronic_codes_survivors": round(float(chc[m][yy == 0].mean()), 2),
        }
    for tag in ("submitted_cohort", "24h_cohort"):
        v_ = ac[tag]["auroc_random_subset_same_rate"]
        ac[tag]["random_subset_min_max_mean"] = [min(v_), max(v_), round(float(np.mean(v_)), 4)]
    ac["chronic_retention_rate"] = round(rate, 4)
    ac["note"] = (f"{N_RAND} random draws; the submitted paper used 3 draws whose values depend on "
                  "row order, so its 0.6237/0.6258/0.6279 are not expected to reproduce exactly")
    sc = ac["submitted_cohort"]
    gate(gates, "chronic-only count", sc["auroc_chronic_only"], PUBLISHED["full_chronic_only"])
    gate(gates, "non-chronic count", sc["auroc_nonchronic"], PUBLISHED["full_nonchronic"])
    gate(gates, "C(chronic | total)", sc["cond_chronic_given_total"],
         PUBLISHED["full_cond_chronic_given_total"])
    gate(gates, "C(total | chronic)", sc["cond_total_given_chronic"],
         PUBLISHED["full_cond_total_given_chronic"])
    out["appendix_c"] = ac
    log(f"  Appendix C, 24-h cohort: {ac['24h_cohort']}")

    # ---- ICD version
    icd = {}
    v = pd.to_numeric(df.icdv, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    for ver in (9, 10):
        m = v == ver
        icd[f"icd{ver}_submitted"] = {"n": int(m.sum()), "auroc": round(rank_auroc(cnt[m], y[m]), 4)}
        ma = m & anchor
        icd[f"icd{ver}_24h_cohort"] = {"n": int(ma.sum()), "deaths": int(y[ma].sum()),
                                       "auroc": round(rank_auroc(cnt[ma], y[ma]), 4),
                                       "mean_codes": round(float(cnt[ma].mean()), 2)}
        n_, a_ = PUBLISHED[f"icd{ver}"]
        gate(gates, f"ICD-{ver} n", icd[f"icd{ver}_submitted"]["n"], n_)
        gate(gates, f"ICD-{ver} AUROC", icd[f"icd{ver}_submitted"]["auroc"], a_)
    out["icd_version"] = icd
    log(f"  ICD split: {icd}")
    return anchor, df


# ---------------------------------------------------------------- E2 ---------
def run_e2(con, M, NOTE, anchor, df, out, skip_poe=False, only=None):
    log("E2: other admission-level counts on the 24-h cohort")
    con.register("anchor_ids", pd.DataFrame({"hadm_id": df.hadm_id.values[anchor]}))
    con.execute("CREATE OR REPLACE TEMP TABLE ca AS "
                "SELECT c.* FROM c JOIN anchor_ids USING (hadm_id)")
    base = con.execute("SELECT hadm_id, subject_id, y, n_dx FROM ca ORDER BY hadm_id").df()
    y = base.y.values.astype(int)
    g = base.subject_id.values

    specs = [
        ("procedure codes (distinct)", f"{M}/hosp/procedures_icd.csv.gz",
         "COUNT(DISTINCT t.icd_code)",
         "COUNT(DISTINCT t.icd_code) FILTER (WHERE TRY_CAST(t.chartdate AS DATE) <= CAST(ca.t24 AS DATE))",
         "date resolution only: chartdate <= date(t0 + 24 h)"),
        ("prescriptions (rows)", f"{M}/hosp/prescriptions.csv.gz",
         "COUNT(*)",
         "COUNT(*) FILTER (WHERE TRY_CAST(t.starttime AS TIMESTAMP) <= ca.t24)", "starttime"),
        ("prescriptions (distinct drugs)", f"{M}/hosp/prescriptions.csv.gz",
         "COUNT(DISTINCT t.drug)",
         "COUNT(DISTINCT t.drug) FILTER (WHERE TRY_CAST(t.starttime AS TIMESTAMP) <= ca.t24)", "starttime"),
        ("microbiology specimens (distinct)", f"{M}/hosp/microbiologyevents.csv.gz",
         "COUNT(DISTINCT t.micro_specimen_id)",
         "COUNT(DISTINCT t.micro_specimen_id) FILTER (WHERE COALESCE(TRY_CAST(t.charttime AS TIMESTAMP), "
         "TRY_CAST(t.chartdate AS TIMESTAMP)) <= ca.t24)", "charttime, else chartdate"),
    ]
    specs.append(("laboratory tests (rows)", f"{M}/hosp/labevents.csv.gz", "COUNT(*)",
                  "COUNT(*) FILTER (WHERE TRY_CAST(t.charttime AS TIMESTAMP) <= ca.t24)", "charttime"))
    if not skip_poe:
        specs.append(("provider orders, POE (rows)", f"{M}/hosp/poe.csv.gz", "COUNT(*)",
                      "COUNT(*) FILTER (WHERE TRY_CAST(t.ordertime AS TIMESTAMP) <= ca.t24)", "ordertime"))
    if NOTE:
        specs.append(("radiology notes", f"{NOTE}/note/radiology.csv.gz", "COUNT(*)",
                      "COUNT(*) FILTER (WHERE TRY_CAST(t.charttime AS TIMESTAMP) <= ca.t24)", "charttime"))

    if only:
        specs = [sp for sp in specs if only.lower() in sp[0].lower()]
    res = {"cohort": {"n": int(len(y)), "deaths": int(y.sum())},
           "window_note": ("24-hour versions keep rows whose event time (start, order or chart "
                           "time) is at or before t0 + 24 h; procedures use chartdate <= date(t0 + 24 h), "
                           "which reaches the end of that calendar day; microbiology falls back to "
                           "chartdate when charttime is missing"),
           "ci_method": f"{N_BOOT}-replicate subject-cluster bootstrap of the oriented AUROC",
           "reference: distinct diagnosis codes (no timestamp; whole admission only)": {
               "auroc": round(roc_auc_score(y, base.n_dx.values), 4),
               "ci95": boot_ci(base.n_dx.values.astype(float), y, g)}}
    for name, path, agg_all, agg_24, timecol in specs:
        if not Path(path).exists():
            log(f"  skip {name}: file not found")
            res[name] = "file not found"
            continue
        t1 = time.time()
        q = (f"SELECT ca.hadm_id, {agg_all} AS n_all, {agg_24} AS n_24 "
             f"FROM {csv_src(path)} t JOIN ca ON TRY_CAST(t.hadm_id AS BIGINT) = ca.hadm_id "
             f"GROUP BY ca.hadm_id")
        try:
            agg = con.execute(q).df()
        except duckdb.Error as e:
            log(f"  {name}: strict parse failed ({type(e).__name__}); retrying with strict_mode=false")
            q2 = q.replace("all_varchar=true)", "all_varchar=true, strict_mode=false)")
            agg = con.execute(q2).df()
        m = base[["hadm_id"]].merge(agg, on="hadm_id", how="left").fillna(0)
        rec = {"time_column": timecol, "seconds": round(time.time() - t1, 1)}
        for col, lab in (("n_all", "whole admission"), ("n_24", "up to t0 + 24 h")):
            s = m[col].values.astype(float)
            a = roc_auc_score(y, s)
            rec[lab] = {"coverage": round(float((s > 0).mean()), 4),
                        "auroc": round(float(a), 4),
                        "oriented": round(float(max(a, 1 - a)), 4),
                        "ci95_oriented": boot_ci(s, y, g)}
        rec["whole_minus_24h_oriented"] = round(
            rec["whole admission"]["oriented"] - rec["up to t0 + 24 h"]["oriented"], 4)
        res[name] = rec
        log(f"  {name}: whole {rec['whole admission']['oriented']}  "
            f"24h {rec['up to t0 + 24 h']['oriented']}  ({rec['seconds']} s)")
    out["other_counts_24h_cohort"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-only", action="store_true")
    ap.add_argument("--skip-poe", action="store_true")
    ap.add_argument("--only", default=None, help="run only E2 counts whose name contains this")
    ap.add_argument("--out", default="extended_checks.json")
    a = ap.parse_args()
    M = os.environ.get("MIMIC_IV_PATH")
    if not M:
        raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root (…/mimiciv/3.1)")
    M = M.replace("\\", "/").rstrip("/")
    NOTE = os.environ.get("MIMIC_IV_NOTE_PATH")
    NOTE = NOTE.replace("\\", "/").rstrip("/") if NOTE else None
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")
    out = {"corpus": "MIMIC-IV v3.1", "aggregate_only": True}
    anchor, df = run_e1(con, M, out)
    if not a.e1_only:
        run_e2(con, M, NOTE, anchor, df, out, skip_poe=a.skip_poe, only=a.only)
    failed = [x["gate"] for x in out["gates"] if not x["pass"]]
    out["gates_failed"] = failed
    out["wall_seconds"] = round(time.time() - t0, 1)
    dest = OUT / a.out
    dest.write_text(json.dumps(out, indent=2, default=str))
    log(f"{len(out['gates']) - len(failed)}/{len(out['gates'])} reproduction gates passed"
        + (f"; FAILED: {failed}" if failed else ""))
    log(f"DONE in {out['wall_seconds']} s -> {dest}")


if __name__ == "__main__":
    main()
