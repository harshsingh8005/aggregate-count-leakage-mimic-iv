"""Isolation test of every laboratory feature in the shipped set.

Rank AUROC of each first-24h laboratory aggregate as a sole predictor of
in-hospital mortality, on the admissions where that feature is available,
plus a head-to-head against the distinct-code count on their common support.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.
"""
import duckdb, numpy as np, json
import os
import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
LABS={"creatinine":(50912,),"wbc":(51301,51300),"hemoglobin":(51222,),"platelet":(51265,),
      "glucose":(50931,50809),"sodium":(50983,),"potassium":(50971,),"bicarbonate":(50882,),
      "bun":(51006,),"lactate":(50813,)}
CLIP={"creatinine":(0,30),"wbc":(0,200),"hemoglobin":(0,30),"platelet":(0,2000),
      "glucose":(0,2000),"sodium":(80,200),"potassium":(0,15),"bicarbonate":(0,60),
      "bun":(0,300),"lactate":(0,40)}
allids=sorted({i for v in LABS.values() for i in v})

con=duckdb.connect(); con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY","6GB") + "'; SET threads=" + os.environ.get("DUCKDB_THREADS","4") + ";")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE base AS
WITH f AS (SELECT hadm_id,subject_id,intime,
             ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
           FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
d AS (SELECT hadm_id,COUNT(DISTINCT icd_code) n_dx
      FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
SELECT f.hadm_id,f.intime,a.hospital_expire_flag y,COALESCE(d.n_dx,0) n_dx,p.anchor_age age
FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
       JOIN read_csv_auto('{M}/hosp/patients.csv.gz') p ON p.subject_id=f.subject_id
       LEFT JOIN d USING (hadm_id) WHERE f.rn=1
""")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE labs AS
SELECT l.hadm_id, l.itemid, AVG(l.valuenum) v
FROM read_csv_auto('{M}/hosp/labevents.csv.gz',
      strict_mode=false, ignore_errors=true,
      columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                'specimen_id':'BIGINT','itemid':'BIGINT','order_provider_id':'VARCHAR',
                'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                'comments':'VARCHAR'}}) l
JOIN base b USING (hadm_id)
WHERE l.itemid IN ({','.join(map(str,allids))}) AND l.valuenum IS NOT NULL
  AND l.charttime >= b.intime AND l.charttime <= b.intime + INTERVAL 24 HOUR
GROUP BY l.hadm_id,l.itemid
""")
print("  labs aggregated")

def auroc(x,yy):
    x=np.asarray(x,float); yy=np.asarray(yy,int)
    o=np.argsort(x,kind="mergesort"); xs=x[o]; r=np.empty(len(x)); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    n1=int(yy.sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)

b=con.execute("SELECT hadm_id,y,n_dx,age FROM base").df()
rows=[]
print("\n=== ISOLATION TEST: every lab in the shipped set ===")
print(f"  {'feature':14} {'coverage':>9} {'AUROC':>8} {'oriented':>9}")
for name,ids in LABS.items():
    lo,hi=CLIP[name]
    q=con.execute(f"SELECT hadm_id, AVG(v) v FROM labs WHERE itemid IN ({','.join(map(str,ids))}) "
                  f"AND v BETWEEN {lo} AND {hi} GROUP BY hadm_id").df()
    m=b.merge(q,on="hadm_id",how="inner")
    if len(m)<1000 or m.y.sum()<50: continue
    a=auroc(m.v.values,m.y.values); rows.append((name,len(m)/len(b),a,max(a,1-a)))
    print(f"  {name:14} {len(m)/len(b):8.1%} {a:8.4f} {max(a,1-a):9.4f}")
a_cnt=auroc(b.n_dx.values,b.y.values); a_age=auroc(b.age.values,b.y.values)
print(f"  {'--count--':14} {1.0:8.1%} {a_cnt:8.4f} {a_cnt:9.4f}")
print(f"  {'--age--':14} {1.0:8.1%} {a_age:8.4f} {a_age:9.4f}")

best=max(rows,key=lambda r:r[3])
print(f"\n  strongest legitimate single feature: {best[0]} at {best[3]:.4f}")
lo,hi=CLIP[best[0]]
q=con.execute(f"SELECT hadm_id, AVG(v) v FROM labs WHERE itemid IN ({','.join(map(str,LABS[best[0]]))}) "
              f"AND v BETWEEN {lo} AND {hi} GROUP BY hadm_id").df()
m=b.merge(q,on="hadm_id",how="inner")
print(f"  head-to-head on the {len(m):,} admissions with a first-24h {best[0]}:")
print(f"    count {auroc(m.n_dx.values,m.y.values):.4f}   vs   {best[0]} {auroc(m.v.values,m.y.values):.4f}")
json.dump({"labs":[{"f":r[0],"cov":round(r[1],4),"auroc":round(r[2],4),"oriented":round(r[3],4)} for r in rows],
           "count":round(float(a_cnt),4),"age":round(float(a_age),4),
           "head_to_head":{"feature":best[0],"n":int(len(m)),
                           "count":round(float(auroc(m.n_dx.values,m.y.values)),4),
                           "rival":round(float(auroc(m.v.values,m.y.values)),4)}},
          open(_os.path.join(_OUT,"isolation_sweep.json"),"w"),indent=2)
print("\nwrote outputs/isolation_sweep.json")
