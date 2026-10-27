"""Full-corpus W1 cohort builder for the leakage audit (READ-ONLY analysis).

Reuses the AUDITED loader's own helpers for the small tables (admissions,
patients, icustays, Charlson prior-history), and reimplements ONLY the two
heavy windowed aggregations (chartevents vitals, labevents labs) in DuckDB so
the 3.3 GB / 2.5 GB gzip files can be streamed on a memory-constrained box.

The DuckDB aggregation mirrors loader._window_means exactly: same itemid maps,
same 24h inclusive window, same per-signal plausibility clip, same Fahrenheit
fold. validate_against_loader() proves column-for-column equivalence on the
demo before this is trusted on the credentialed corpus.

HARD PHI RULE: this module returns/holds only aggregate feature frames keyed on
opaque ids used transiently in-memory. It NEVER prints row-level values, ids,
or ICD codes. Callers must keep it that way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Import the audited loader module directly so cohort + Charlson use its code.
_W1_SRC = Path(__file__).resolve().parent.parent / "reproducer_src"
sys.path.insert(0, str(_W1_SRC))
from src import loader as L  # noqa: E402


def _base_cohort(root: Path) -> pd.DataFrame:
    """Replicate loader.load_w1_cohort base assembly (lines ~244-276) exactly."""
    admissions = L._read(
        root, "hosp", "admissions",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "admission_type", "admission_location", "insurance",
                 "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime"],
    )
    patients = L._read(root, "hosp", "patients",
                       usecols=["subject_id", "gender", "anchor_age"])
    icustays = L._read(root, "icu", "icustays",
                       usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                       parse_dates=["intime", "outtime"])
    icustays = icustays.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    cohort = (icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
              .merge(patients, on="subject_id", how="left"))
    cohort["_win_end"] = cohort["intime"] + L.OBS_WINDOW
    # Charlson prior-history via the AUDITED loader function.
    dx = L._read(root, "hosp", "diagnoses_icd",
                 usecols=["hadm_id", "icd_code", "icd_version"])
    history = L._charlson_history(dx, admissions, cohort)
    cohort["charlson_history"] = cohort["hadm_id"].map(history).fillna(0).astype(int)
    return cohort


def _duck_window_means(con, gzip_path, key, cohort_win, itemid_map, plausible,
                       time_col="charttime"):
    """DuckDB replica of loader._window_means for one heavy table.

    key: 'stay_id' (vitals) or 'hadm_id' (labs). cohort_win is a DataFrame with
    columns [key, 'intime', '_win_end']. time_col is 'charttime' or 'storetime'
    (Task 5-critical uses 'storetime'). Returns a frame indexed by key with one
    mean column per signal, matching _window_means semantics.
    """
    con.register("win", cohort_win[[key, "intime", "_win_end"]])
    # itemid -> signal, and per-signal (lo,hi) using the loader's own lookup.
    id_rows, rng_rows = [], []
    for name, ids in itemid_map.items():
        lo, hi = plausible.get(name, plausible.get(name.rstrip("_cf"), (-np.inf, np.inf)))
        for iid in ids:
            id_rows.append((iid, name))
        rng_rows.append((name, float(lo), float(hi)))
    con.register("idmap", pd.DataFrame(id_rows, columns=["itemid", "signal"]))
    con.register("rng", pd.DataFrame(rng_rows, columns=["signal", "lo", "hi"]))
    q = f"""
    WITH ev AS (
        SELECT e.{key} AS k, e.itemid, e.valuenum, e.{time_col} AS t
        FROM read_csv_auto('{gzip_path}', types={{'{time_col}':'TIMESTAMP','valuenum':'DOUBLE'}}) e
        WHERE e.{key} IS NOT NULL AND e.valuenum IS NOT NULL
    )
    SELECT ev.k AS {key}, m.signal, avg(ev.valuenum) AS v
    FROM ev
    JOIN win w ON w.{key} = ev.k
    JOIN idmap m ON m.itemid = ev.itemid
    JOIN rng r ON r.signal = m.signal
    WHERE ev.t >= w.intime AND ev.t <= w._win_end
      AND ev.valuenum >= r.lo AND ev.valuenum <= r.hi
    GROUP BY ev.k, m.signal
    """
    long = con.execute(q).df()
    if long.empty:
        return pd.DataFrame(index=pd.Index([], name=key))
    wide = long.pivot(index=key, columns="signal", values="v")
    wide.columns.name = None
    return wide


def build_cohort_frame(root, con=None, time_col="charttime"):
    """Build the full W1 feature frame (loader-equivalent) via DuckDB heavies.

    Returns (frame, meta) where frame has the loader's canonical columns plus
    'n_diagnoses' (distinct icd_code count for the CURRENT hadm_id, the pre-fix
    leaked feature) and 'subject_id' for grouped splitting. meta holds counts.
    """
    root = Path(root)
    close = False
    if con is None:
        con = duckdb.connect(); close = True
    cohort = _base_cohort(root)

    # Vitals (chartevents) keyed on stay_id.
    vwin = cohort[["stay_id", "intime", "_win_end"]]
    ce = str(root / "icu" / "chartevents.csv.gz")
    vit = _duck_window_means(con, ce, "stay_id", vwin, L.VITAL_ITEMIDS, L._PLAUSIBLE,
                             time_col=time_col)
    # Fold Fahrenheit into a single Celsius temperature (loader lines 292-300).
    temp_c = vit.get("temp_c"); temp_f = vit.get("temp_f")
    temp = temp_c.copy() if temp_c is not None else pd.Series(index=vit.index, dtype=float)
    if temp_f is not None:
        temp = temp.fillna((temp_f - 32.0) * 5.0 / 9.0)
    temp = temp.clip(*L._PLAUSIBLE["temp"])
    vit = vit.drop(columns=[c for c in ("temp_c", "temp_f") if c in vit.columns])
    vit["temperature"] = temp
    cohort = cohort.set_index("stay_id").join(vit, how="left")

    # Labs (labevents) keyed on hadm_id, windowed against ICU intime.
    lwin = cohort.reset_index()[["hadm_id", "intime", "_win_end"]]
    le = str(root / "hosp" / "labevents.csv.gz")
    lab = _duck_window_means(con, le, "hadm_id", lwin, L.LAB_ITEMIDS, L._PLAUSIBLE,
                             time_col=time_col)
    cohort = cohort.merge(lab, on="hadm_id", how="left")

    # Pre-fix leaked feature: distinct icd_code count for the CURRENT hadm_id.
    di = str(root / "hosp" / "diagnoses_icd.csv.gz")
    ndx = con.execute(
        f"SELECT hadm_id, count(DISTINCT icd_code) AS n_diagnoses "
        f"FROM read_csv_auto('{di}') GROUP BY hadm_id").df()
    cohort = cohort.merge(ndx, on="hadm_id", how="left")
    cohort["n_diagnoses"] = cohort["n_diagnoses"].fillna(0).astype(int)

    ordered = [c for c in L._canonical_columns() if c in cohort.columns]
    keep = ["subject_id", "hadm_id", "n_diagnoses"] + ordered
    frame = cohort.reset_index(drop=True)[keep].copy()
    frame[L.LABEL_COL] = frame[L.LABEL_COL].astype(int)
    meta = {"n": int(len(frame)),
            "n_pos": int(frame[L.LABEL_COL].sum()),
            "n_subjects": int(frame["subject_id"].nunique())}
    if close:
        con.close()
    return frame, meta


def validate_against_loader(demo_root):
    """Prove the DuckDB build matches loader.load_w1_cohort on the demo."""
    ref = L.load_w1_cohort(demo_root).frame.reset_index(drop=True)
    mine, _ = build_cohort_frame(demo_root)
    mine = mine[ref.columns.tolist()].reset_index(drop=True)
    # Sort both identically for row-aligned comparison (label + all features).
    cols = ref.columns.tolist()
    ref_s = ref.sort_values(cols).reset_index(drop=True)
    mine_s = mine.sort_values(cols).reset_index(drop=True)
    report = {}
    for c in cols:
        a, b = ref_s[c], mine_s[c]
        if pd.api.types.is_float_dtype(a):
            close = np.allclose(a.fillna(-999), b.fillna(-999), atol=1e-6, rtol=1e-6)
        else:
            close = a.equals(b)
        report[c] = bool(close)
    report["_shape_ref"] = ref.shape
    report["_shape_mine"] = mine.shape
    return report


if __name__ == "__main__":
    import os
    demo = os.environ.get("MIMIC_IV_DEMO_PATH")
    if not demo:
        raise SystemExit("set MIMIC_IV_DEMO_PATH to the open-access 100-patient demo root")
    demo = Path(demo)
    rep = validate_against_loader(demo)
    ok = all(v for k, v in rep.items() if not k.startswith("_"))
    print("VALIDATION", "PASS" if ok else "FAIL")
    for k, v in rep.items():
        print(f"  {k}: {v}")
