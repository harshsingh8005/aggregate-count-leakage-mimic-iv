"""Missing comparator statistics for the paper revision.

Rank AUROC (Mann-Whitney, tie-corrected) for single features on the paper's
cohort. For a single predictor AUROC is invariant to any strictly monotone
transform, so a model fit adds nothing over the rank statistic.

AGGREGATE OUTPUT ONLY. No row-level values, ids, or codes printed or written.
"""
import duckdb, json, numpy as np

import os
import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
con = duckdb.connect()
con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY","6GB") + "'; SET threads=" + os.environ.get("DUCKDB_THREADS","4") + ";")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE cohort AS
WITH first_icu AS (
  SELECT hadm_id, subject_id, intime,
         ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
  FROM read_csv_auto('{M}/icu/icustays.csv.gz')
),
dx AS (
  SELECT hadm_id,
         COUNT(DISTINCT icd_code) AS n_diagnoses,
         MAX(icd_version) AS icd_max, MIN(icd_version) AS icd_min
  FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz')
  GROUP BY hadm_id
)
SELECT f.hadm_id, f.subject_id,
       a.hospital_expire_flag AS y,
       COALESCE(d.n_diagnoses, 0) AS n_dx,
       date_diff('minute', a.admittime, a.dischtime)/1440.0 AS los_days,
       p.anchor_age AS age,
       d.icd_min, d.icd_max
FROM first_icu f
JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
JOIN read_csv_auto('{M}/hosp/patients.csv.gz')   p ON p.subject_id = f.subject_id
LEFT JOIN dx d USING (hadm_id)
WHERE f.rn = 1
""")

n, ns, nd = con.execute("SELECT COUNT(*), COUNT(DISTINCT subject_id), SUM(y) FROM cohort").fetchone()
print(f"cohort: n={n:,}  subjects={ns:,}  deaths={nd:,}  prevalence={nd/n:.4%}")
print("  (paper: 85,242 / 65,366 / 9,475 / 11.12%)\n")

def auroc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, int)
    order = np.argsort(x, kind="mergesort"); xs = x[order]
    ranks = np.empty(len(x), float); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j+1] == xs[i]: j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)

df = con.execute("SELECT y, n_dx, los_days, age, icd_min, icd_max FROM cohort").df()
y = df.y.values
out = {"cohort": {"n": int(n), "subjects": int(ns), "deaths": int(nd), "prevalence": round(nd/n, 6)}}

print("=== SINGLE-FEATURE RANK AUROC (whole cohort, no split) ===")
out["auroc_single_feature"] = {}
for name, col in [("distinct_code_count","n_dx"), ("length_of_stay","los_days"), ("age","age")]:
    a = float(auroc(df[col].values, y)); out["auroc_single_feature"][name] = round(a,4)
    print(f"  {name:22} {a:.4f}")
json.dump(out, open(_os.path.join(_OUT,"comparators.json"),"w"), indent=2)
print("\nwrote outputs/comparators.json")
