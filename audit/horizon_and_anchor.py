"""Ascertainment-window sensitivity, and the stay-length argument at a valid anchor.

Two things a reviewer will ask for, both of which the earlier drafts deferred.

1. FIXED HORIZON. The published label is in-hospital death with ascertainment
   running the whole stay, so a duration-linked feature is advantaged by
   construction. Recoding the label as death within H days of admission
   measures how much of the count's discrimination depends on that open-ended
   window. It is a large effect and it is reported here whether or not it
   flatters the paper.

2. CONDITIONAL CONCORDANCE AT THE 24-HOUR ANCHOR. On the published cohort,
   stratifying by stay length stratifies on a collider: for the great majority
   of decedents the discharge time *is* the death time, so a stay-length
   stratum mixes time-to-event for cases with time-to-censoring for controls.
   Restricting to admissions alive and inpatient at 24 hours removes the
   admissions whose label precedes the feature window and makes the
   stratification interpretable. The script reports the fraction of decedents
   whose dischtime equals their deathtime so the reader can size the problem.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/horizon_and_anchor.py
"""
import json
import os
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


def rank_auroc(score, y):
    s = np.asarray(score, float)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    srt = np.sort(s)
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
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def conditional_c(score, strata, y):
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
    return num / den


def strata_of(x, k):
    e = np.quantile(x, np.linspace(0, 1, k + 1))
    e[0] -= 1
    e[-1] += 1
    return np.clip(np.digitize(x, e) - 1, 0, k - 1)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")
    df = con.execute(f"""
        WITH f AS (SELECT hadm_id, subject_id, intime,
                          ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
                   FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
             d AS (SELECT hadm_id, COUNT(DISTINCT icd_code) n_dx
                   FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
        SELECT f.hadm_id, a.hospital_expire_flag y, COALESCE(d.n_dx, 0) n_dx,
               date_diff('minute', a.admittime, a.dischtime) / 1440.0 los,
               date_diff('minute', a.admittime, a.deathtime) / 1440.0 death_from_adm,
               date_diff('minute', f.intime,   a.deathtime) / 1440.0 death_from_icu,
               date_diff('minute', f.intime,   a.dischtime) / 1440.0 los_from_icu,
               date_diff('minute', a.deathtime, a.dischtime)         disch_minus_death
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
               LEFT JOIN d USING (hadm_id) WHERE f.rn = 1""").df()
    y = df.y.values.astype(int)
    cnt = df.n_dx.values.astype(float)
    los = df.los.values.astype(float)
    out = {"n": int(len(df)), "n_deaths": int(y.sum())}

    # --- how bad is the collider? dischtime == deathtime for decedents
    dd = df.disch_minus_death.values
    dec = (y == 1) & df.death_from_adm.notna().values
    same = np.sum(np.abs(dd[dec]) < 1e-9)
    out["decedents_with_dischtime_eq_deathtime"] = int(same)
    out["decedents_with_dischtime_eq_deathtime_frac"] = round(float(same / dec.sum()), 4)

    # --- 1. fixed horizon
    out["fixed_horizon"] = {}
    for H in (7, 14, 30):
        yh = (df.death_from_adm.notna().values & (df.death_from_adm.values <= H)).astype(int)
        out["fixed_horizon"][f"{H}d"] = {
            "events": int(yh.sum()),
            "auroc_count": round(float(rank_auroc(cnt, yh)), 4)}
    out["fixed_horizon"]["open_ended"] = {
        "events": int(y.sum()), "auroc_count": round(float(rank_auroc(cnt, y)), 4)}

    # --- 2. conditional concordance, published cohort vs alive-at-24h
    # same predicate the anchor script uses: alive AND still inpatient at 24 h
    d_icu = np.where(df.death_from_icu.notna().values, df.death_from_icu.values, np.inf)
    alive = ~((d_icu <= 1.0) | (df.los_from_icu.values <= 1.0))
    for tag, m in (("published", np.ones(len(df), bool)), ("alive_at_24h", alive)):
        yy, cc, ll = y[m], cnt[m], los[m]
        rec = {"n": int(m.sum()), "deaths": int(yy.sum()),
               "marginal_count": round(float(rank_auroc(cc, yy)), 4),
               "marginal_los": round(float(rank_auroc(ll, yy)), 4)}
        for k in (5, 500):
            rec[f"cond_count_given_los_{k}"] = round(
                float(conditional_c(cc, strata_of(ll, k), yy)), 4)
            rec[f"cond_los_given_los_{k}"] = round(
                float(conditional_c(ll, strata_of(ll, k), yy)), 4)
        out[tag] = rec

    for k, v in out.items():
        print(f"  {k}: {v}")
    (OUT / "horizon_and_anchor.json").write_text(json.dumps(out, indent=2))
    print("  wrote outputs/horizon_and_anchor.json")


if __name__ == "__main__":
    main()
