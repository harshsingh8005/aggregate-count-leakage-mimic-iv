"""Subject-cluster bootstrap intervals for the vitals isolation sweep.

Extends sweep_intervals.py to the vital-sign features: 95% intervals for the
oriented rank AUROC of the vitals at or near the 0.65 bar, recomputed alongside
the near-bar laboratory features and the distinct-code count on the identical
cohort so all eight interval estimates share one footing.

Reuses vitals_sweep.py for the cohort (first ICU stay per hadm_id, tie-broken
on stay_id) and the chartevents extraction constants, and sweep_intervals.py
for boot_ci, so every number is computed by the same code that produced the
point estimates and the laboratory intervals.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/vitals_intervals.py
"""
import json
import os
import time
from pathlib import Path

import duckdb

import sweep_intervals as si
import vitals_sweep as vs

# stdout only: vitals_sweep.log belongs to the sweep, not the interval pass
vs.log = print

OUT = Path(__file__).resolve().parent.parent / "outputs" / "vitals_intervals.json"

VITAL_TARGETS = ("resp_rate", "sbp", "heart_rate")
# isolation_sweep.py's recipe for the near-bar labs: single itemid each,
# inclusive 24h window, clip applied to the per-admission mean
LAB_TARGETS = {"creatinine": ((50912,), (0, 30)), "bicarbonate": ((50882,), (0, 60)),
               "bun": ((51006,), (0, 300)), "lactate": ((50813,), (0, 40))}
LAB_NOTE = "isolation_sweep.py recipe (inclusive 24h window, clipped admission mean)"

LE_COLS = ("{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',"
           "'specimen_id':'BIGINT','itemid':'BIGINT','order_provider_id':'VARCHAR',"
           "'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',"
           "'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',"
           "'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',"
           "'comments':'VARCHAR'}")


def atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def report(name, entry):
    pt = entry.get("auroc_oriented", entry.get("auroc"))
    lo, hi = entry["lo95"], entry["hi95"]
    side = ("CROSSES 0.65" if lo < 0.65 < hi
            else "entirely above 0.65" if lo >= 0.65 else "entirely below 0.65")
    print(f"  {name:20} {pt:.4f} [{lo:.4f}, {hi:.4f}]  cov {entry['coverage']:.1%}  {side}")


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4;")
    b = vs.build_cohort(con)
    con.register("coh", b[["hadm_id", "intime", "y"]])
    y = b.y.values.astype(int)
    g = b.subject_id.values
    out = {"seed": si.SEED, "n_boot": si.N_BOOT, "n": int(len(b)),
           "design": "resamples subject_id with replacement"}

    # Vitals: one chartevents pass for all targets. None of the targets needs
    # the Fahrenheit fold; extending to temperature would require it.
    case = " ".join(f"WHEN e.itemid IN ({','.join(map(str, vs.VITALS[v]))}) THEN '{v}'"
                    for v in VITAL_TARGETS)
    clip = " ".join(f"WHEN vital = '{v}' THEN v BETWEEN {vs.CLIP[v][0]} AND {vs.CLIP[v][1]}"
                    for v in VITAL_TARGETS)
    allids = sorted({i for v in VITAL_TARGETS for i in vs.VITALS[v]})
    q = con.execute(f"""
    WITH ev AS (
        SELECT e.hadm_id, e.charttime, e.valuenum AS v,
               CASE {case} END AS vital
        FROM read_csv_auto('{vs.M}/icu/chartevents.csv.gz',
              strict_mode=false, ignore_errors=true, columns={vs.CE_COLS}) e
        WHERE e.itemid IN ({','.join(map(str, allids))})
          AND e.valuenum IS NOT NULL AND e.hadm_id IS NOT NULL
    )
    SELECT b.hadm_id, ev.vital, AVG(ev.v) AS v
    FROM ev
    JOIN coh b ON b.hadm_id = ev.hadm_id
    WHERE ev.charttime >= b.intime AND ev.charttime <= b.intime + INTERVAL 24 HOUR
      AND CASE {clip} END
    GROUP BY b.hadm_id, ev.vital
    ORDER BY b.hadm_id, ev.vital
    """).df()
    for name in VITAL_TARGETS:
        sub = q[q.vital == name]
        v = b.hadm_id.map(sub.set_index("hadm_id").v)
        m = v.notna().values
        pt, lo, hi = si.boot_ci(v[m].values.astype(float), y[m], g[m], oriented=True)
        out[name] = {"auroc_oriented": pt, "lo95": lo, "hi95": hi,
                     "coverage": round(float(m.mean()), 4),
                     "itemids": list(vs.VITALS[name])}
        report(name, out[name])
        atomic_json(OUT, out)

    # Near-bar labs: one labevents pass.
    allids = sorted(i for ids, _ in LAB_TARGETS.values() for i in ids)
    q = con.execute(f"""
    SELECT b.hadm_id, l.itemid, AVG(l.valuenum) AS v
    FROM read_csv_auto('{vs.M}/hosp/labevents.csv.gz',
          strict_mode=false, ignore_errors=true, columns={LE_COLS}) l
    JOIN coh b ON b.hadm_id = l.hadm_id
    WHERE l.itemid IN ({','.join(map(str, allids))}) AND l.valuenum IS NOT NULL
      AND l.charttime >= b.intime AND l.charttime <= b.intime + INTERVAL 24 HOUR
    GROUP BY b.hadm_id, l.itemid
    ORDER BY b.hadm_id, l.itemid
    """).df()
    for name, (ids, (lo_c, hi_c)) in LAB_TARGETS.items():
        sub = q[q.itemid.isin(ids) & q.v.between(lo_c, hi_c)]
        v = b.hadm_id.map(sub.set_index("hadm_id").v)
        m = v.notna().values
        pt, lo, hi = si.boot_ci(v[m].values.astype(float), y[m], g[m], oriented=True)
        out[name] = {"auroc_oriented": pt, "lo95": lo, "hi95": hi,
                     "coverage": round(float(m.mean()), 4),
                     "itemids": list(ids), "extraction": LAB_NOTE}
        report(name, out[name])
        atomic_json(OUT, out)

    # The leak, un-oriented, exactly as sweep_intervals.py records it.
    ndx = con.execute(f"""
    SELECT b.hadm_id, COALESCE(d.n_dx, 0) AS n_dx
    FROM coh b
    LEFT JOIN (SELECT hadm_id, COUNT(DISTINCT icd_code) n_dx
               FROM read_csv_auto('{vs.M}/hosp/diagnoses_icd.csv.gz')
               GROUP BY hadm_id) d USING (hadm_id)
    ORDER BY b.hadm_id
    """).df()
    v = b.hadm_id.map(ndx.set_index("hadm_id").n_dx).values.astype(float)
    pt, lo, hi = si.boot_ci(v, y, g)
    out["distinct_code_count"] = {
        "auroc": pt, "lo95": lo, "hi95": hi, "coverage": 1.0,
        "extraction": "COUNT(DISTINCT icd_code) per hadm_id from diagnoses_icd"}
    report("distinct_code_count", out["distinct_code_count"])
    atomic_json(OUT, out)
    print(f"  wrote outputs/vitals_intervals.json  total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
