"""Isolation test of the comorbidity index — the row Table 3 was missing.

The manuscript claims that a prior-admission Charlson index recovers less
discrimination than the current-admission count it replaces. Earlier drafts
inferred that from a nineteen-feature ablation (arm (a) vs arm (c)). This
script measures it directly: the index as a sole predictor, scored by rank
AUROC on the same cohort.

The index is rebuilt with the audited reproducer's own `charlson_index` and its
own strict predicate — a prior admission counts only if its `dischtime` falls
before the current `admittime` — so the distribution reported here should
reproduce the one in Appendix F (36,319 non-zero, 42.6%, mean 1.83, max 22).
That agreement is the check that this reimplementation is faithful.

AGGREGATE OUTPUT ONLY. No row-level value or identifier is written.

Usage:
    export MIMIC_IV_PATH=/path/to/physionet.org/files/mimiciv/3.1
    python audit/comorbidity_isolation.py
"""
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reproducer_src"))
from src.charlson import charlson_index  # noqa: E402

M = os.environ.get("MIMIC_IV_PATH")
if not M:
    raise SystemExit("set MIMIC_IV_PATH to the MIMIC-IV v3.1 root")
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


def rank_auroc(score, y):
    """Tie-corrected Mann-Whitney U / (n1 n0)."""
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
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='" + os.environ.get("DUCKDB_MEMORY", "6GB")
                + "'; SET threads=" + os.environ.get("DUCKDB_THREADS", "4") + ";")

    coh = con.execute(f"""
        WITH f AS (SELECT hadm_id,
                          ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
                   FROM read_csv_auto('{M}/icu/icustays.csv.gz'))
        SELECT a.hadm_id, a.subject_id, a.hospital_expire_flag y, a.admittime
        FROM f JOIN read_csv_auto('{M}/hosp/admissions.csv.gz') a USING (hadm_id)
        WHERE f.rn = 1""").df()
    adm = con.execute(f"""SELECT hadm_id, subject_id, admittime, dischtime
        FROM read_csv_auto('{M}/hosp/admissions.csv.gz')
        ORDER BY subject_id, admittime""").df()
    dx = con.execute(f"""SELECT hadm_id, icd_code, icd_version
        FROM read_csv_auto('{M}/hosp/diagnoses_icd.csv.gz')""").df()

    codes = {h: list(zip(g.icd_code, g.icd_version))
             for h, g in dx.groupby("hadm_id", sort=False)}
    want = set(coh.hadm_id)
    prior = {}
    for _, g in adm.groupby("subject_id", sort=False):
        hs, at, dt = g.hadm_id.values, g.admittime.values, g.dischtime.values
        for i in range(len(hs)):
            if hs[i] not in want:
                continue
            acc = set()
            for j in range(i):
                if pd.notna(dt[j]) and dt[j] < at[i]:
                    acc.update(codes.get(hs[j], ()))
            prior[hs[i]] = charlson_index(acc) if acc else 0

    ch = coh.hadm_id.map(prior).fillna(0).values.astype(float)
    y = coh.y.values.astype(int)

    out = {
        "n": int(len(coh)),
        "n_deaths": int(y.sum()),
        "nonzero": int((ch > 0).sum()),
        "nonzero_frac": round(float((ch > 0).mean()), 4),
        "mean": round(float(ch.mean()), 4),
        "max": int(ch.max()),
        "auroc_solo": round(float(rank_auroc(ch, y)), 4),
        "auroc_on_nonzero_subset": round(float(rank_auroc(ch[ch > 0], y[ch > 0])), 4),
        "note": ("distribution should match Appendix F (36,319 / 42.6% / 1.83 / 22); "
                 "agreement is the fidelity check on this reimplementation"),
    }
    print(f"  n={out['n']} nonzero={out['nonzero']} ({out['nonzero_frac']:.1%}) "
          f"mean={out['mean']} max={out['max']}")
    print(f"  comorbidity index SOLO rank AUROC = {out['auroc_solo']:.4f}")
    (OUT / "comorbidity_isolation.json").write_text(json.dumps(out, indent=2))
    print("  wrote outputs/comorbidity_isolation.json")


if __name__ == "__main__":
    main()
