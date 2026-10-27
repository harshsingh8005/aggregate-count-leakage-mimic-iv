"""Isolation test of the four categorical admission descriptors.

Rank AUROC of each categorical as a sole predictor of in-hospital mortality,
scored by a cross-fitted per-level death rate so the encoding cannot leak the
label it is being tested against. Subject-grouped folds, matching the protocol
used for every other number in the paper.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.
"""
import json
import os
import time

import duckdb
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

M = os.environ["MIMIC_IV_PATH"]
SEED, N_SPLITS = 42, 5
CATS = ("gender", "admission_type", "admission_location", "insurance")

t0 = time.time()
con = duckdb.connect()
con.execute("SET memory_limit='2GB'; SET threads=2;")
df = con.execute(f"""
WITH f AS (SELECT hadm_id, subject_id, intime,
             ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime, stay_id) rn
           FROM read_csv_auto('{M}/icu/icustays.csv.gz'))
SELECT f.hadm_id, f.subject_id,
       a.hospital_expire_flag AS y,
       a.admission_type, a.admission_location, a.insurance,
       p.gender
FROM f
JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
JOIN read_csv_auto('{M}/hosp/patients.csv.gz') p ON p.subject_id = f.subject_id
WHERE f.rn = 1
ORDER BY f.hadm_id
""").df()

n, deaths = len(df), int(df.y.sum())
print(f"  cohort n={n:,}  subjects={df.subject_id.nunique():,}  deaths={deaths:,} ({deaths/n:.2%})")
assert n == 85_242, f"COHORT MISMATCH: {n}"
assert deaths == 9_475, f"DEATHS MISMATCH: {deaths}"


def rank_auroc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, int)
    o = np.argsort(x, kind="mergesort"); xs = x[o]
    r = np.empty(len(x)); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


y = df.y.values
groups = df.subject_id.values
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(df, y, groups))

rows = []
print(f"\n  {'feature':22} {'levels':>7} {'coverage':>9} {'AUROC':>8} {'oriented':>9}")
for c in CATS:
    col = df[c].astype("string")
    cov = float(col.notna().mean())
    oof = np.full(n, np.nan)
    for tr, te in folds:
        gm = y[tr].mean()
        rate = df.iloc[tr].assign(_y=y[tr]).groupby(c, observed=True)["_y"].mean()
        oof[te] = col.iloc[te].map(rate).astype(float).fillna(gm).values
    a = rank_auroc(oof, y)
    rows.append({"f": c, "levels": int(col.nunique()), "cov": round(cov, 4),
                 "auroc": round(float(a), 4), "oriented": round(float(max(a, 1 - a)), 4)})
    print(f"  {c:22} {col.nunique():7d} {cov:8.1%} {a:8.4f} {max(a, 1-a):9.4f}")

out = {"design": f"cross-fitted per-level death rate, StratifiedGroupKFold "
                 f"n_splits={N_SPLITS} seed={SEED} grouped on subject_id",
       "n": n, "deaths": deaths, "categoricals": rows,
       "wall_s": round(time.time() - t0, 1)}
print(json.dumps(out, indent=2))
json.dump(out, open("/tmp/categorical_sweep.json", "w"), indent=2)
