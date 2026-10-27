"""Task 4 (first-admission Charlson audit) + Task 5 storetime exclusion counts.

AGGREGATE-ONLY. Reuses the audited loader's _charlson_history via cohort_build.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproducer_src"))
from cohort_build import _base_cohort  # noqa: E402
import src.loader as L  # noqa: E402

ROOT = Path(os.environ.get("MIMIC_IV_PATH", ""))  # mimiciv/3.1
LAB_IDS = [i for ids in L.LAB_ITEMIDS.values() for i in ids]
VIT_IDS = [i for ids in L.VITAL_ITEMIDS.values() for i in ids]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
_os.makedirs(_OUT, exist_ok=True)

def main():
    con = duckdb.connect(); con.execute("PRAGMA threads=" + os.environ.get("DUCKDB_THREADS","4"))
    out = {}

    log("TASK 4: build base cohort (audited charlson) + first-admission flag")
    cohort = _base_cohort(ROOT)  # has subject_id, hadm_id, admittime, charlson_history
    # First admission per subject by earliest admittime, over ALL admissions.
    adm = L._read(ROOT, "hosp", "admissions", usecols=["subject_id", "hadm_id", "admittime"],
                  parse_dates=["admittime"])
    first = adm.sort_values("admittime").drop_duplicates("subject_id", keep="first")
    first_hadms = set(first["hadm_id"].astype(int))
    is_first = cohort["hadm_id"].astype(int).isin(first_hadms)
    ch = cohort["charlson_history"].to_numpy()
    n_first = int(is_first.sum())
    n_first_nonzero = int(((ch > 0) & is_first.to_numpy()).sum())
    out["task4"] = {
        "cohort_n": int(len(cohort)),
        "n_first_admissions_in_cohort": n_first,
        "n_first_admissions_with_nonzero_charlson": n_first_nonzero,
        "charlson_history_nonzero_overall": int((ch > 0).sum()),
        "charlson_history_zero_overall": int((ch == 0).sum()),
        "charlson_history_max": int(ch.max()),
        "charlson_history_mean": float(ch.mean()),
        "predicate": "loader.py:221 keep prior iff prior_hadm != cur_hadm and dischtime < admittime (strict)",
    }
    log(f"  first-admissions in cohort={n_first}, of which nonzero charlson={n_first_nonzero} "
        f"(expected 0 if temporal logic correct)")

    # Task 5b: rows in the 24h CHARTTIME window whose STORETIME > win_end
    # (charted in-window but not yet stored at the 24h point -> would be excluded
    # by storetime-based filtering). Denominator = rows in charttime window.
    log("TASK 5b: storetime exclusion counts (heavy scans)")
    win_v = cohort[["stay_id", "intime", "_win_end"]]
    win_l = cohort[["hadm_id", "intime", "_win_end"]]
    con.register("win_v", win_v); con.register("win_l", win_l)

    def excl(path, key, idlist, winname):
        ids = ",".join(str(i) for i in idlist)
        q = f"""
        SELECT count(*) AS in_charttime_window,
               sum(CASE WHEN e.storetime > w._win_end THEN 1 ELSE 0 END) AS excluded_by_storetime
        FROM read_csv_auto('{path}', types={{'charttime':'TIMESTAMP','storetime':'TIMESTAMP'}}) e
        JOIN {winname} w ON w.{key} = e.{key}
        WHERE e.itemid IN ({ids}) AND e.valuenum IS NOT NULL
          AND e.charttime >= w.intime AND e.charttime <= w._win_end
        """
        return con.execute(q).df().iloc[0].to_dict()

    lab = excl(str(ROOT / "hosp" / "labevents.csv.gz"), "hadm_id", LAB_IDS, "win_l")
    log(f"  labevents: in-window={int(lab['in_charttime_window'])} "
        f"excluded_by_storetime={int(lab['excluded_by_storetime'])}")
    che = excl(str(ROOT / "icu" / "chartevents.csv.gz"), "stay_id", VIT_IDS, "win_v")
    log(f"  chartevents: in-window={int(che['in_charttime_window'])} "
        f"excluded_by_storetime={int(che['excluded_by_storetime'])}")
    out["task5b"] = {
        "labevents": {k: int(v) for k, v in lab.items()},
        "chartevents": {k: int(v) for k, v in che.items()},
        "definition": "rows with charttime in [intime, intime+24h] but storetime > intime+24h",
    }
    Path(_OUT, "task4_task5b.json").write_text(json.dumps(out, indent=2))
    log("DONE -> task4_task5b.json")


if __name__ == "__main__":
    main()
