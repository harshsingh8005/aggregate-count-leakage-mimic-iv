"""Per-itemid split of the resp_rate isolation result.

The shipped resp_rate feature pools two chartevents itemids: 220210
(Respiratory Rate) and 224690 (Respiratory Rate (Total), charted essentially
only on ventilated patients). This scores each itemid separately as a sole
predictor of in-hospital mortality — same cohort, window, clip, and AUROC
implementation as the pooled sweep — and adds a head-to-head on the admissions
where both are present, so the pooled number can be attributed between the
value signal and the ventilation-linked measurement process.

Reuses vitals_sweep.py for the cohort, the extraction constants, and auroc, so
every number is computed by the code that produced the pooled point estimate.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/resp_itemid_split.py
"""
import json
import os
import time
from pathlib import Path

import duckdb

import vitals_sweep as vs

# stdout only: vitals_sweep.log belongs to the sweep, not this split
vs.log = print

OUT = Path(__file__).resolve().parent.parent / "outputs" / "resp_itemid_split.json"

ITEMIDS = {220210: "Respiratory Rate", 224690: "Respiratory Rate (Total)"}


def atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def score(m):
    a = float(vs.auroc(m.v.values, m.y.values))
    return round(a, 4), round(max(a, 1 - a), 4)


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4;")
    b = vs.build_cohort(con)
    con.register("coh", b[["hadm_id", "intime", "y"]])

    lo, hi = vs.CLIP["resp_rate"]
    q = con.execute(f"""
    WITH ev AS (
        SELECT e.hadm_id, e.itemid, e.charttime, e.valuenum AS v
        FROM read_csv_auto('{vs.M}/icu/chartevents.csv.gz',
              strict_mode=false, ignore_errors=true, columns={vs.CE_COLS}) e
        WHERE e.itemid IN ({','.join(map(str, ITEMIDS))})
          AND e.valuenum IS NOT NULL AND e.hadm_id IS NOT NULL
    )
    SELECT b.hadm_id, ev.itemid, AVG(ev.v) AS v, COUNT(*) AS n_obs
    FROM ev
    JOIN coh b ON b.hadm_id = ev.hadm_id
    WHERE ev.charttime >= b.intime AND ev.charttime <= b.intime + INTERVAL 24 HOUR
      AND ev.v BETWEEN {lo} AND {hi}
    GROUP BY b.hadm_id, ev.itemid
    ORDER BY b.hadm_id, ev.itemid
    """).df()

    out = {"n": int(len(b)), "deaths": int(b.y.sum()), "clip": [lo, hi],
           "itemids": {}, "common_support": {}}
    merged, have = {}, {}
    print(f"\n  {'itemid':>8} {'coverage':>9} {'admissions':>11} {'deaths':>7} "
          f"{'rate':>7} {'AUROC':>8} {'oriented':>9} {'rows':>12}")
    for iid, label in ITEMIDS.items():
        g = q[q.itemid == iid]
        m = b.merge(g[["hadm_id", "v"]], on="hadm_id", how="inner")
        merged[iid], have[iid] = m, set(m.hadm_id)
        adm, dth = len(m), int(m.y.sum())
        a, o = score(m)
        out["itemids"][str(iid)] = {
            "label": label, "coverage": round(adm / len(b), 4),
            "admissions": adm, "deaths": dth,
            "death_rate": round(dth / adm, 4), "rows": int(g.n_obs.sum()),
            "auroc": a, "oriented": o}
        e = out["itemids"][str(iid)]
        print(f"  {iid:>8} {e['coverage']:8.1%} {adm:>11,} {dth:>7,} "
              f"{e['death_rate']:6.1%} {a:8.4f} {o:9.4f} {e['rows']:>12,}")

    both = have[220210] & have[224690]
    cs = {"admissions": len(both)}
    print(f"\n  common support: {len(both):,} admissions with both itemids")
    for iid in ITEMIDS:
        m = merged[iid][merged[iid].hadm_id.isin(both)]
        a, o = score(m)
        cs[f"auroc_{iid}"] = a
        cs[f"oriented_{iid}"] = o
        cs["deaths"] = int(m.y.sum())
        print(f"    {iid}: AUROC {a:.4f} oriented {o:.4f}")
    cs["death_rate"] = round(cs["deaths"] / cs["admissions"], 4)
    out["common_support"] = cs
    print(f"    deaths {cs['deaths']:,} ({cs['death_rate']:.1%})")

    atomic_json(OUT, out)
    print(f"\n  wrote outputs/resp_itemid_split.json  total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
