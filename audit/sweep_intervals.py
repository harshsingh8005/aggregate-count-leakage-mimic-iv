"""Subject-cluster bootstrap intervals for the isolation sweep.

Table 3 draws a 0.65 flagging bar across a column of point estimates. A bar
without intervals cannot say which side of it a feature is on, and the answer
matters: it decides how many legitimate features the rule we propose would flag.

This computes 95% intervals by resampling *subjects* with replacement, matching
the design used everywhere else in the paper, for the distinct-code count and
for the laboratory features that sit near the bar.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/sweep_intervals.py
"""
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reproducer_src"))
from src.loader import LAB_ITEMIDS  # noqa: E402

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
N_BOOT = 400
# the features that decide the flag count: the leak, and the labs near 0.65
NEAR_BAR = ("bicarbonate", "creatinine", "bun", "lactate")


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
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, y, groups, oriented=False, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    by_g = {g: np.asarray(ix) for g, ix in by_g.items()}
    uniq = np.fromiter(by_g.keys(), dtype=np.asarray(groups).dtype)
    stats = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in samp])
        a = rank_auroc(score[idx], y[idx])
        if not np.isnan(a):
            stats.append(max(a, 1 - a) if oriented else a)
    pt = rank_auroc(score, y)
    if oriented:
        pt = max(pt, 1 - pt)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(pt), 4), round(float(lo), 4), round(float(hi), 4)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")
    coh = con.execute(f"""
        WITH f AS (SELECT hadm_id, subject_id, intime,
                          ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
                   FROM read_csv_auto('{M}/icu/icustays.csv.gz')),
             d AS (SELECT hadm_id, COUNT(DISTINCT icd_code) n_dx
                   FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz') GROUP BY hadm_id)
        SELECT f.hadm_id, f.subject_id, f.intime, a.hospital_expire_flag y,
               COALESCE(d.n_dx, 0) n_dx
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
               LEFT JOIN d USING (hadm_id) WHERE f.rn = 1
        ORDER BY f.hadm_id""").df()
    # DuckDB does not guarantee row order without ORDER BY, and the subject
    # cluster bootstrap indexes by first appearance, so the interval would not
    # reproduce run to run without it.
    y = coh.y.values.astype(int)
    g = coh.subject_id.values
    out = {"seed": SEED, "n_boot": N_BOOT, "n": int(len(coh)),
           "design": "resamples subject_id with replacement"}

    pt, lo, hi = boot_ci(coh.n_dx.values.astype(float), y, g)
    out["distinct_code_count"] = {"auroc": pt, "lo95": lo, "hi95": hi,
                                  "coverage": 1.0}
    print(f"  distinct-code count  {pt:.4f} [{lo:.4f}, {hi:.4f}]")
    # write now: the laboratory pass below needs a second scan over labevents
    # and may exceed a small machine's patience. The count's interval is quoted
    # in the paper and must not depend on that pass completing.
    (OUT / "sweep_intervals.json").write_text(json.dumps(out, indent=2))

    items = {k: v for k, v in LAB_ITEMIDS.items() if k in NEAR_BAR}
    con.register("coh", coh[["hadm_id", "intime"]])
    for name, ids in items.items():
        idlist = ",".join(str(i) for i in ids)
        lab = con.execute(f"""
            SELECT l.hadm_id, AVG(l.valuenum) v
            FROM read_csv_auto('{M}/hosp/labevents.csv.gz',
                               strict_mode=false, ignore_errors=true) l
            JOIN coh c ON c.hadm_id = l.hadm_id
            WHERE l.itemid IN ({idlist}) AND l.valuenum IS NOT NULL
              AND l.charttime >= c.intime
              AND l.charttime <  c.intime + INTERVAL 24 HOUR
            GROUP BY l.hadm_id""").df()
        v = coh.hadm_id.map(lab.set_index("hadm_id").v)
        m = v.notna().values
        pt, lo, hi = boot_ci(v[m].values.astype(float), y[m], g[m], oriented=True)
        out[name] = {"auroc_oriented": pt, "lo95": lo, "hi95": hi,
                     "coverage": round(float(m.mean()), 4)}
        print(f"  {name:20} {pt:.4f} [{lo:.4f}, {hi:.4f}]  cov {m.mean():.1%}"
              f"  {'CROSSES 0.65' if lo < 0.65 < hi else ''}")
        (OUT / "sweep_intervals.json").write_text(json.dumps(out, indent=2))

    (OUT / "sweep_intervals.json").write_text(json.dumps(out, indent=2))
    print("  wrote outputs/sweep_intervals.json")


if __name__ == "__main__":
    main()
