"""Isolation test of the seven first-24h vital-sign aggregates.

Rank AUROC of each vital-sign mean (first 24h from ICU intime, itemid variants
pooled, physiologic clip applied before aggregation) as a sole predictor of
in-hospital mortality, on the admissions where that vital is available. The
AUROC implementation is copied verbatim from isolation_sweep.py so the numbers
are computed identically to the laboratory features.

chartevents.csv.gz is 3.5 GB, so each logical vital is a separate resumable
pass: results land in a per-vital manifest via atomic replace, and a re-run
skips anything already done. A hard wall-clock cap aborts cleanly with the
remainder marked pending.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np

M = os.environ["MIMIC_IV_PATH"]
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(_OUT, exist_ok=True)
RESULTS = os.path.join(_OUT, "vitals_sweep.json")
MANIFEST = os.path.join(_OUT, "vitals_sweep.manifest.json")
LOG = os.path.join(_OUT, "vitals_sweep.log")

VITALS = {"heart_rate": (220045,), "sbp": (220050, 220179), "dbp": (220051, 220180),
          "mbp": (220052, 220181), "resp_rate": (220210, 224690), "spo2": (220277,),
          "temperature": (223762, 223761)}
CLIP = {"heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 200), "mbp": (10, 250),
        "resp_rate": (1, 80), "spo2": (10, 100), "temperature": (25, 45)}
FAHRENHEIT_ITEMID = 223761
MAX_WALL_S = 6 * 3600

CE_COLS = ("{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT',"
           "'caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP',"
           "'itemid':'BIGINT','value':'VARCHAR','valuenum':'DOUBLE',"
           "'valueuom':'VARCHAR','warning':'BIGINT'}")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def auroc(x, yy):
    x = np.asarray(x, float); yy = np.asarray(yy, int)
    o = np.argsort(x, kind="mergesort"); xs = x[o]; r = np.empty(len(x)); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    n1 = int(yy.sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def build_cohort(con):
    """First ICU stay per hadm_id, ordered by (intime, stay_id)."""
    b = con.execute(f"""
    WITH f AS (SELECT hadm_id, subject_id, intime,
                 ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime, stay_id) rn
               FROM read_csv_auto('{M}/icu/icustays.csv.gz'))
    SELECT f.hadm_id, f.subject_id, f.intime, a.hospital_expire_flag AS y
    FROM f
    JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
    JOIN read_csv_auto('{M}/hosp/patients.csv.gz') p ON p.subject_id = f.subject_id
    WHERE f.rn = 1
    ORDER BY f.hadm_id
    """).df()
    n, subjects, deaths = len(b), int(b.subject_id.nunique()), int(b.y.sum())
    log(f"cohort n={n:,} subjects={subjects:,} deaths={deaths:,}")
    assert n == 85_242, f"COHORT MISMATCH: n={n}"
    assert subjects == 65_366, f"COHORT MISMATCH: subjects={subjects}"
    assert deaths == 9_475, f"COHORT MISMATCH: deaths={deaths}"
    return b


def run_vital(con, b, name):
    ids = VITALS[name]
    lo, hi = CLIP[name]
    q = con.execute(f"""
    WITH ev AS (
        SELECT e.hadm_id, e.charttime,
               CASE WHEN e.itemid = {FAHRENHEIT_ITEMID}
                    THEN (e.valuenum - 32.0) * 5.0 / 9.0
                    ELSE e.valuenum END AS v
        FROM read_csv_auto('{M}/icu/chartevents.csv.gz',
              strict_mode=false, ignore_errors=true, columns={CE_COLS}) e
        WHERE e.itemid IN ({','.join(map(str, ids))})
          AND e.valuenum IS NOT NULL AND e.hadm_id IS NOT NULL
    )
    SELECT b.hadm_id, AVG(ev.v) AS v, COUNT(*) AS n_obs
    FROM ev
    JOIN coh b ON b.hadm_id = ev.hadm_id
    WHERE ev.charttime >= b.intime AND ev.charttime <= b.intime + INTERVAL 24 HOUR
      AND ev.v BETWEEN {lo} AND {hi}
    GROUP BY b.hadm_id
    ORDER BY b.hadm_id
    """).df()
    rows = int(q.n_obs.sum())
    m = b.merge(q[["hadm_id", "v"]], on="hadm_id", how="inner")
    cov = len(m) / len(b)
    a = float(auroc(m.v.values, m.y.values))
    assert 0.0 < cov <= 1.0, f"{name}: coverage {cov} outside (0,1]"
    assert 0.0 <= a <= 1.0, f"{name}: auroc {a} outside [0,1]"
    return {"cov": round(cov, 4), "auroc": round(a, 4),
            "oriented": round(max(a, 1 - a), 4), "rows": rows}


def write_results(manifest):
    vitals = [{"f": v, "cov": manifest[v]["cov"], "auroc": manifest[v]["auroc"],
               "oriented": manifest[v]["oriented"]}
              for v in VITALS if manifest.get(v, {}).get("status") == "done"]
    atomic_json(RESULTS, {"n": 85_242, "deaths": 9_475, "vitals": vitals})


def main(argv):
    t_start = time.time()
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    requested = [a for a in argv if not a.startswith("--")] or list(VITALS)
    for name in requested:
        if name not in VITALS:
            raise SystemExit(f"unknown vital: {name}")

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4;")
    b = build_cohort(con)
    if "--cohort-only" in argv:
        return
    con.register("coh", b[["hadm_id", "intime", "y"]])

    for name in requested:
        if manifest.get(name, {}).get("status") == "done":
            log(f"{name}: already done, skipping")
            continue
        if time.time() - t_start > MAX_WALL_S:
            for rest in requested:
                if manifest.get(rest, {}).get("status") != "done":
                    manifest[rest] = {"status": "pending", "itemids": list(VITALS[rest])}
            atomic_json(MANIFEST, manifest)
            log("WALL CAP BREACHED: wrote completed vitals, rest marked pending")
            sys.exit(1)
        manifest[name] = {"status": "running", "itemids": list(VITALS[name])}
        atomic_json(MANIFEST, manifest)
        log(f"{name}: scanning itemids {list(VITALS[name])}")
        t0 = time.time()
        r = run_vital(con, b, name)
        wall = round(time.time() - t0, 1)
        manifest[name] = {"status": "done", "itemids": list(VITALS[name]),
                          "rows": r["rows"], "wall_s": wall, "cov": r["cov"],
                          "auroc": r["auroc"], "oriented": r["oriented"]}
        atomic_json(MANIFEST, manifest)
        write_results(manifest)
        log(f"{name}: done cov={r['cov']:.4f} auroc={r['auroc']:.4f} "
            f"oriented={r['oriented']:.4f} rows={r['rows']:,} wall_s={wall}")

    done = sum(1 for v in VITALS if manifest.get(v, {}).get("status") == "done")
    log(f"run complete: {done}/{len(VITALS)} vitals done, "
        f"total wall_s={round(time.time() - t_start, 1)}")


if __name__ == "__main__":
    main(sys.argv[1:])
