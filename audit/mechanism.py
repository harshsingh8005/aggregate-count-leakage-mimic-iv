"""Interrogate the length-of-stay mechanism claim. AGGREGATE OUTPUT ONLY."""
import duckdb, json, numpy as np
from scipy import stats

import os
M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
con = duckdb.connect(); con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY","6GB") + "'; SET threads=" + os.environ.get("DUCKDB_THREADS","4") + ";")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE cohort AS
WITH f AS (SELECT hadm_id, subject_id, intime,
             ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
           FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
d AS (SELECT hadm_id, COUNT(DISTINCT icd_code) n_dx, MIN(icd_version) v0, MAX(icd_version) v1
      FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
SELECT a.hospital_expire_flag y, COALESCE(d.n_dx,0) n_dx,
       date_diff('minute', a.admittime, a.dischtime)/1440.0 los, d.v0, d.v1
FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
       LEFT JOIN d USING (hadm_id) WHERE f.rn=1
""")
df = con.execute("SELECT * FROM cohort").df()
y = df.y.values; out = {}

def auroc(x, yy):
    x=np.asarray(x,float); yy=np.asarray(yy,int)
    o=np.argsort(x,kind="mergesort"); xs=x[o]; r=np.empty(len(x)); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    n1=int(yy.sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)

print("=== 1. LENGTH OF STAY BY OUTCOME ===")
for lab, m in [("survivors", y==0), ("deaths", y==1)]:
    v = df.los[m]
    print(f"  {lab:10} n={m.sum():>6,}  mean={v.mean():6.2f}d  median={np.median(v):5.2f}d  Q25={np.percentile(v,25):5.2f}  Q75={np.percentile(v,75):6.2f}")
out["los_by_outcome"] = {"survivor_median": round(float(np.median(df.los[y==0])),3),
                         "death_median":    round(float(np.median(df.los[y==1])),3)}

print("\n=== 2. SPEARMAN rho(count, LOS) — pooled vs stratified ===")
rp = stats.spearmanr(df.n_dx, df.los).statistic
r0 = stats.spearmanr(df.n_dx[y==0], df.los[y==0]).statistic
r1 = stats.spearmanr(df.n_dx[y==1], df.los[y==1]).statistic
print(f"  pooled      {rp:.4f}   (paper reports 0.529)")
print(f"  survivors   {r0:.4f}")
print(f"  deaths      {r1:.4f}")
out["spearman"] = {"pooled":round(float(rp),4),"survivors":round(float(r0),4),"deaths":round(float(r1),4)}

print("\n=== 3. AUROC of the count WITHIN length-of-stay strata ===")
print("     (if LOS-proxying drives it, within-stratum AUROC should collapse)")
qs = np.percentile(df.los,[20,40,60,80]); strata=[]
edges=[-np.inf,*qs,np.inf]
for i in range(5):
    m=(df.los>edges[i])&(df.los<=edges[i+1])
    if m.sum()>500 and 20 < y[m].sum() < m.sum()-20:
        a=auroc(df.n_dx[m].values,y[m])
        print(f"  LOS {edges[i]:6.2f}-{edges[i+1]:6.2f}d  n={m.sum():>6,}  deaths={y[m].sum():>5,}  AUROC={a:.4f}")
        strata.append(round(float(a),4))
out["auroc_count_within_los_strata"]=strata
print(f"  --> pooled AUROC(count) = {auroc(df.n_dx.values,y):.4f}")

print("\n=== 4. CAP MASS AT 39, BY CLASS  (Appendix B) ===")
for lab,m in [("survivors",y==0),("deaths",y==1)]:
    c=(df.n_dx[m]==39).sum()
    print(f"  {lab:10} at cap: {c:>6,} / {m.sum():>6,} = {c/m.sum():.3%}")
out["cap_mass"]={"survivor_frac":round(float((df.n_dx[y==0]==39).sum()/ (y==0).sum()),5),
                 "death_frac":round(float((df.n_dx[y==1]==39).sum()/ (y==1).sum()),5)}


import os as _os
_dest = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "outputs", "mechanism.json")
json.dump(out, open(_dest, "w"), indent=2)
print("\nwrote outputs/mechanism.json")
# ICD-version stratification lives in icd.py, which writes outputs/icd.json.
