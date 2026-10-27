"""Controls for the stay-length objection, and the pre-admission decomposition.

Three things, all of which replace weaker arguments in earlier drafts.

1. NULL BEHAVIOUR. Section 4 argues that conditioning on stay length raises the
   count's concordance rather than collapsing it. That is only informative if
   the same conditioning collapses a genuine stay-length proxy, so we condition
   stay length on itself and confirm it moves toward 0.5.

2. INCREMENTAL VALUE. An earlier draft reported Spearman rho between the count
   and a monotone-free recoding of stay length, and read a low value as
   independence. That statistic is vacuous: rho between anything monotone in
   stay length and a non-monotone function of stay length is near zero by
   construction, and stay length scores about the same against the recoding as
   the count does. We replace it with two incremental-value measures, which are
   what the claim actually needs: conditional concordance of the count within
   recoding bins, and out-of-fold logistic AUROC with and without the count.

   Folds are grouped by subject, not assigned by row position, so the result
   does not depend on the order the corpus happens to come back in.

3. IS THE SIGNAL RECOVERABLE COMORBIDITY? Restricting the count to codes
   mapping into Quan-Charlson categories was originally read as measuring how
   much of the signal is standing burden recorded late. That reading does not
   survive three checks reported here: the non-Charlson complement scores
   higher, a random subset of code rows at the same retention rate scores almost
   as well, and conditional on the total count the chronic subset adds little.
   The signal is coding volume. Note also that Quan-Charlson is not a
   pre-admission construct - it contains acute presentations, and 31.4% of
   admissions have a principal diagnosis mapping into it.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/stay_length_controls.py
"""
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reproducer_src"))
from src.charlson import CHARLSON_ICD9, CHARLSON_ICD10  # noqa: E402

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
N_BINS = 20


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
    """Pair-weighted concordance within strata."""
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
        SELECT f.hadm_id, f.subject_id, a.hospital_expire_flag y,
               COALESCE(d.n_dx, 0) n_dx,
               date_diff('minute', a.admittime, a.dischtime) / 1440.0 los
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
               LEFT JOIN d USING (hadm_id) WHERE f.rn = 1""").df()
    y = df.y.values.astype(int)
    cnt = df.n_dx.values.astype(float)
    los = df.los.values.astype(float)
    g = df.subject_id.values

    # --- subject-grouped folds
    uniq = np.unique(g)
    rng = np.random.default_rng(SEED)
    fmap = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    fold = np.array([fmap[s] for s in g])

    # --- 1. null behaviour: does conditioning collapse a real stay-length proxy?
    out = {"seed": SEED, "n": int(len(df))}
    out["auroc_los_marginal"] = round(float(rank_auroc(los, y)), 4)
    for k in (5, 500):
        e = np.quantile(los, np.linspace(0, 1, k + 1))
        e[0] -= 1
        e[-1] += 1
        strata = np.clip(np.digitize(los, e) - 1, 0, k - 1)
        out[f"cond_los_given_los_{k}"] = round(float(conditional_c(los, strata, y)), 4)

    # --- 2. monotone-free recoding of stay length, cross-fitted
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
    out["auroc_recoded_los"] = round(float(rank_auroc(rec, y)), 4)
    # NB an earlier version conditioned on `b`, the LOS quantile bin, which is a
    # monotone binning of stay length and therefore the k=20 point of the curve
    # above -- not an independent control. Condition on the recoding itself.
    out["cond_count_given_los_bins_k20"] = round(float(conditional_c(cnt, b, y)), 4)
    out["cond_count_given_recoded_values"] = round(
        float(conditional_c(cnt, rec, y)), 4)
    out["auroc_count_marginal"] = round(float(rank_auroc(cnt, y)), 4)

    def oof(X):
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            p[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return rank_auroc(p, y)

    out["oof_recoded_los"] = round(float(oof(rec.reshape(-1, 1))), 4)
    out["oof_recoded_los_plus_count"] = round(
        float(oof(np.column_stack([rec, cnt]))), 4)

    # --- 3. how much of the count is pre-admission chronic burden
    dxr = con.execute(f"""SELECT hadm_id, icd_code, icd_version
        FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz')""").df()
    dxr = dxr[dxr.hadm_id.isin(set(df.hadm_id))]
    p9 = tuple(sorted({p for v in CHARLSON_ICD9.values() for p in v}))
    p10 = tuple(sorted({p for v in CHARLSON_ICD10.values() for p in v}))
    chronic = [str(c).strip().startswith(p10 if v == 10 else p9)
               for c, v in zip(dxr.icd_code, dxr.icd_version)]
    dxr_all = dxr
    chronic_all = chronic
    dxr = dxr[np.asarray(chronic)]
    per = dxr.groupby("hadm_id").icd_code.nunique()
    chc = df.hadm_id.map(per).fillna(0).values.astype(float)
    out["auroc_chronic_only_count"] = round(float(rank_auroc(chc, y)), 4)
    out["mean_chronic_codes_deaths"] = round(float(chc[y == 1].mean()), 2)
    out["mean_chronic_codes_survivors"] = round(float(chc[y == 0].mean()), 2)
    # The complement, which an excess-concordance "share" would require to be
    # the remainder. It is not: the two "shares" sum to well over one, because
    # excess concordance does not decompose additively over a partition of the
    # codes. Reported so the invalid reading cannot be made.
    nonchr = df.hadm_id.map(
        dxr_all[~np.asarray(chronic_all)].groupby("hadm_id").icd_code.nunique()
    ).fillna(0).values.astype(float)
    out["auroc_nonchronic_count"] = round(float(rank_auroc(nonchr, y)), 4)

    # A random subset of code rows at the same retention rate. This is the null
    # the "share" has to beat, and it very nearly matches it -- so the statistic
    # is measuring how many codes were kept, not which ones.
    rate = float(np.mean(chronic_all))
    out["chronic_retention_rate"] = round(rate, 4)
    draws = []
    rng2 = np.random.default_rng(0)
    for _ in range(3):
        keep = rng2.random(len(dxr_all)) < rate
        r = df.hadm_id.map(
            dxr_all[keep].groupby("hadm_id").icd_code.nunique()
        ).fillna(0).values.astype(float)
        draws.append(round(float(rank_auroc(r, y)), 4))
    out["auroc_random_subset_same_rate"] = draws

    # Conditional on the total count, how much does chronic membership add --
    # and conversely. This is the comparison that identifies which is doing the
    # work, and it points at volume.
    out["cond_chronic_given_total"] = round(float(conditional_c(chc, cnt, y)), 4)
    out["cond_total_given_chronic"] = round(float(conditional_c(cnt, chc, y)), 4)

    for k, v in out.items():
        print(f"  {k:38} {v}")
    (OUT / "stay_length_controls.json").write_text(json.dumps(out, indent=2))
    print("  wrote outputs/stay_length_controls.json")


if __name__ == "__main__":
    main()
