"""Anchor validity and stay-length stratification for the aggregate-count leak.

Computes: deaths falling inside the 24h feature window; rank AUROC of the
distinct-code count and of length of stay, as published and restricted to
admissions alive and inpatient at 24h and 48h; conditional concordance of the
count within 1/5/20/100/500 stay-length strata; rank AUROC of a cross-fitted
bin-wise recoding of stay length and its Spearman correlation with the count;
and the shortest-quintile mean code count by outcome.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.
"""
import duckdb, numpy as np, json
from scipy.stats import spearmanr

import os
import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
con=duckdb.connect(); con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY","6GB") + "'; SET threads=" + os.environ.get("DUCKDB_THREADS","4") + ";")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE c AS
WITH f AS (SELECT hadm_id,subject_id,intime,
             ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
           FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
d AS (SELECT hadm_id,COUNT(DISTINCT icd_code) n_dx
      FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
SELECT f.hadm_id, f.subject_id, a.hospital_expire_flag y,
       COALESCE(d.n_dx,0) n_dx,
       date_diff('minute',a.admittime,a.dischtime)/1440.0 los,
       date_diff('minute',f.intime,a.dischtime)/1440.0 los_from_icu,
       CASE WHEN a.deathtime IS NULL THEN NULL
            ELSE date_diff('minute',f.intime,a.deathtime)/1440.0 END AS death_from_icu
FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
       LEFT JOIN d USING (hadm_id) WHERE f.rn=1
       ORDER BY f.hadm_id
""")
df=con.execute("SELECT * FROM c").df()
y=df.y.values.astype(int); ndx=df.n_dx.values.astype(float); los=df.los.values.astype(float)

def auroc(x,yy):
    x=np.asarray(x,float); yy=np.asarray(yy,int)
    o=np.argsort(x,kind="mergesort"); xs=x[o]; r=np.empty(len(x)); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    n1=int(yy.sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)

V={}
print("=== 1. ANCHOR VIOLATION: deaths inside the 24h feature window ===")
dfi=df.death_from_icu.values
for h,lab in [(24,"24h"),(48,"48h")]:
    k=int(np.nansum(dfi <= h/24.0))
    print(f"  deaths within {lab} of ICU intime: {k:,} of {y.sum():,} = {k/y.sum():.2%}")
    V[f"deaths_within_{lab}"]=k

print("\n=== 2. AUROC UNDER A VALID ANCHOR (alive + inpatient at 24h) ===")
alive24 = ~((dfi<=1.0) | (df.los_from_icu.values<=1.0))
alive48 = ~((dfi<=2.0) | (df.los_from_icu.values<=2.0))
for lab,m in [("as published",np.ones(len(y),bool)),("alive at 24h",alive24),("alive at 48h",alive48)]:
    print(f"  {lab:16} n={m.sum():>6,} deaths={y[m].sum():>5,}  AUROC(LOS)={auroc(los[m],y[m]):.4f}  AUROC(count)={auroc(ndx[m],y[m]):.4f}")
    V[lab]={"n":int(m.sum()),"auroc_los":round(float(auroc(los[m],y[m])),4),"auroc_count":round(float(auroc(ndx[m],y[m])),4)}

print("\n=== 3. CONDITIONAL CONCORDANCE over LOS strata (the correct estimand) ===")
for K in [1,5,20,100,500]:
    edges=np.unique(np.percentile(los,np.linspace(0,100,K+1)))
    b=np.clip(np.digitize(los,edges[1:-1]),0,len(edges)-2)
    conc=disc=tie=0.0
    for s in np.unique(b):
        m=b==s; xs=ndx[m]; ys=y[m]
        if ys.sum()==0 or ys.sum()==len(ys): continue
        p=xs[ys==1]; n=xs[ys==0]
        conc+=float((p[:,None]>n[None,:]).sum()); disc+=float((p[:,None]<n[None,:]).sum())
        tie +=float((p[:,None]==n[None,:]).sum())
    cc=(conc+0.5*tie)/(conc+disc+tie)
    print(f"  {K:>3} strata: conditional AUROC = {cc:.4f}")
    V[f"conditional_{K}"]=round(float(cc),4)

print("\n=== 4. rho(count, cross-fitted LOS risk) ===")
# folds are assigned by row position, so the cohort query above carries an
# explicit ORDER BY: DuckDB does not guarantee row order, and without it the
# cross-fitted values below drift between runs.
rng=np.random.default_rng(42); fold=rng.integers(0,5,len(y)); risk=np.empty(len(y))
edges=np.unique(np.percentile(los,np.linspace(0,100,21)))
b=np.clip(np.digitize(los,edges[1:-1]),0,len(edges)-2)
for f_ in range(5):
    tr=fold!=f_; te=fold==f_
    rate={s: y[tr&(b==s)].mean() if (tr&(b==s)).sum()>0 else y[tr].mean() for s in np.unique(b)}
    risk[te]=[rate[s] for s in b[te]]
print(f"  AUROC(recoded LOS, cross-fitted) = {auroc(risk,y):.4f}")
r=spearmanr(ndx,risk).statistic
print(f"  rho(count, recoded LOS risk)     = {r:.4f}")
V["auroc_los_recoded"]=round(float(auroc(risk,y)),4); V["rho_count_losrisk"]=round(float(r),4)

print("\n=== 5. Q1 mean codes on stays <= 3.4 days ===")
q1=los<=np.percentile(los,20)
print(f"  Q1 n={q1.sum():,}  decedents mean={ndx[q1&(y==1)].mean():.2f}  survivors mean={ndx[q1&(y==0)].mean():.2f}")
V["q1"]={"dec":round(float(ndx[q1&(y==1)].mean()),2),"sur":round(float(ndx[q1&(y==0)].mean()),2)}
json.dump(V,open(_os.path.join(_OUT,"anchor_and_stratification.json"),"w"),indent=2,default=str); print("\nwrote outputs/anchor_and_stratification.json")
