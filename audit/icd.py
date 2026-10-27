"""ICD-version stratification + suppressor check. AGGREGATE OUTPUT ONLY."""
import duckdb, json, numpy as np
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
WITH f AS (SELECT hadm_id,intime,ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
           FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
d AS (SELECT hadm_id,COUNT(DISTINCT icd_code) n_dx,MIN(icd_version) v0,MAX(icd_version) v1
      FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
SELECT a.hospital_expire_flag y, COALESCE(d.n_dx,0) n_dx,
       date_diff('minute',a.admittime,a.dischtime)/1440.0 los,
       COALESCE(d.v0,-1) v0, COALESCE(d.v1,-1) v1
FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
       LEFT JOIN d USING (hadm_id) WHERE f.rn=1
""")
df=con.execute("SELECT * FROM c").df()
y=df.y.values.astype(int); ndx=df.n_dx.values.astype(float); los=df.los.values.astype(float)
v0=df.v0.values.astype(int); v1=df.v1.values.astype(int)

def auroc(x,yy):
    x=np.asarray(x,float); yy=np.asarray(yy,int)
    o=np.argsort(x,kind="mergesort"); xs=x[o]; r=np.empty(len(x)); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    n1=int(yy.sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)

out={}
print("=== ICD VERSION STRATIFICATION ===")
for lab,m in [("ICD-9 only",(v0==9)&(v1==9)),("ICD-10 only",(v0==10)&(v1==10)),("mixed/none",~(((v0==9)&(v1==9))|((v0==10)&(v1==10))))]:
    if m.sum()>100:
        a=auroc(ndx[m],y[m])
        print(f"  {lab:12} n={m.sum():>6,}  mean codes={ndx[m].mean():5.2f}  mortality={y[m].mean():6.2%}  AUROC={a:.4f}")
        out[lab]={"n":int(m.sum()),"mean_codes":round(float(ndx[m].mean()),2),
                  "mortality":round(float(y[m].mean()),4),"auroc":round(float(a),4)}

print("\n=== SUPPRESSION CHECK ===")
print(f"  AUROC(count)            {auroc(ndx,y):.4f}")
print(f"  AUROC(LOS)              {auroc(los,y):.4f}   <- below chance")
# residualise the count on LOS by rank, then re-score
from scipy.stats import rankdata
rl=rankdata(los); rn=rankdata(ndx)
beta=np.polyfit(rl,rn,1)[0]; resid=rn-beta*rl
print(f"  AUROC(count | LOS)      {auroc(resid,y):.4f}   <- rank-residualised on LOS")
out["withdrawn_suppression_see_appendix"]={"count":round(float(auroc(ndx,y)),4),"los":round(float(auroc(los,y)),4),
                    "count_resid_los":round(float(auroc(resid,y)),4)}
json.dump(out,open(_os.path.join(_OUT,"icd.json"),"w"),indent=2); print("\nwrote outputs/icd.json")
