import pandas as pd
import numpy as np
import glob
import os

# --------------------------
# Paths (same as before)
# --------------------------
chunk_folder  = "test_biomarker_tmp_outputsFINAL"  # output from 7_test.py
output_file   = "FINAL_Aurum_with_Tests.csv"
enriched_file = "Enriched_Aurum_with_Biomarkers.csv"

# ---------- Load cohort ----------
df = pd.read_csv(enriched_file, dtype=str)
df['indexdate'] = pd.to_datetime(df['indexdate'], format='%Y-%m-%d', errors='coerce')
df['dod']       = pd.to_datetime(df['dod'], errors='coerce')
df['patid']     = df['patid'].astype(str)

# ---------- Chunk loader (NO date filter now) ----------
def load_test_chunks(prefix):
    files = sorted(glob.glob(os.path.join(chunk_folder, f"{prefix}_chunk_*.csv.gz")))
    if not files:
        return pd.DataFrame(columns=["patid","obsdate","value","unit"])
    parts = []
    for f in files:
        x = pd.read_csv(f, dtype=str)
        for c in ("patid","obsdate","value"):
            if c not in x.columns:
                x[c] = pd.NA
        if "unit" not in x.columns:
            x["unit"] = pd.NA
        x = x[["patid","obsdate","value","unit"]]
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    out["patid"] = out["patid"].astype(str)
    out["obsdate"] = pd.to_datetime(out["obsdate"], errors="coerce", dayfirst=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    # Only require a valid date; no start/end restriction
    out = out[out["obsdate"].notna()]
    return out

hdl   = load_test_chunks("hdl")
ldl   = load_test_chunks("ldl")
trig  = load_test_chunks("triglycerides")
hba1c = load_test_chunks("hba1c")
tc    = load_test_chunks("tot_chol")

# =========================================================
# === Functions with the SAME names/behavior as GOLD ======
# =========================================================

def _as_str_lower(s):
    return s.astype(str).str.strip().str.lower()

def _standardize_lipid_units_like_gold(x: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Emulates GOLD behavior by producing a single mmol/L stream (like filtering unit==96 in GOLD).
    - Convert mg/dL → mmol/L
      * chol/HDL/LDL: *0.02586
      * triglycerides: *0.01129
    - If unit missing/opaque and value > 40, treat as mg/dL and convert.
    - Keep plausible range 0..20 mmol/L.
    """
    if x.empty:
        return x
    y = x.copy()
    unit = _as_str_lower(y["unit"])

    mgdl_syn = {"mg/dl","mgdl","mg per dl","mg%/dl","mg%dl"}
    mmol_syn = {"mmol/l","mmol","mmol per l","mmol/litre"}

    factor = 0.01129 if var == "triglycerides" else 0.02586

    is_mgdl = unit.isin(mgdl_syn)
    # convert explicit mg/dL
    y.loc[is_mgdl, "value"] = y.loc[is_mgdl, "value"] * factor

    # heuristic for missing/unknown unit: big values likely mg/dL
    missingish = unit.isna() | (unit == "nan") | (unit == "")
    looks_mgdl = missingish & (y["value"] > 40)
    y.loc[looks_mgdl, "value"] = y.loc[looks_mgdl, "value"] * factor

    # keep plausible mmol/L range
    y = y[(y["value"].notna()) & (y["value"] >= 0) & (y["value"] <= 20)]
    return y

def get_lab_data(test: pd.DataFrame, patient: pd.DataFrame, enttype_ignored, unit_code_like_96, limit, col):
    """
    GOLD signature preserved:
      get_lab_data(test, patient, enttype, unit, limit, col)
    In Aurum we don't have enttype; we pass a placeholder for compatibility.
    Behavior:
      - Standardize to mmol/L stream (≈ unit 96 in GOLD).
      - Filter by value ≤ limit.
      - Pick nearest to indexdate per patid.
      - Merge into patient as columns: <col>_date, <col>
    """
    if test.empty:
        return patient

    # Standardize to mmol/L like GOLD's unit==96 stream
    var_name = {
        "tot_chol":"tot_chol",
        "hdl":"hdl",
        "ldl":"ldl",
        "trigly":"triglycerides"
    }.get(col, col)

    t = _standardize_lipid_units_like_gold(test, var_name)

    # value cap
    t = t[(t["value"].notna()) & (t["value"] <= limit)].copy()
    if t.empty:
        return patient

    # join indexdate & pick nearest
    t = t[t["patid"].isin(patient["patid"])]
    t = t.merge(patient[["patid","indexdate"]], on="patid", how="left")
    t = t.dropna(subset=["indexdate","obsdate"])
    t["gap"] = (t["indexdate"] - t["obsdate"]).abs().dt.days
    t = t.loc[t.groupby(["patid","indexdate"])["gap"].idxmin()].copy()

    t = t.rename(columns={"obsdate": f"{col}_date", "value": col})
    out = patient.merge(t[["patid","indexdate", f"{col}_date", col]],
                        on=["patid","indexdate"], how="left")
    return out

def get_hba1c_data(patient: pd.DataFrame, test: pd.DataFrame):
    """
    GOLD logic replicated:
      - Convert to % (NGSP) using:
          * if unit is % → keep
          * if unit is IFCC mmol/mol → % = 0.09148*mmol/mol + 2.152
          * if unit missing: values 2–20 → treat as %; otherwise treat as mmol/mol then convert
      - Keep 2–20%
      - Nearest to indexdate
      - Merge as hba1c_perc and hba1c_date
    """
    if test.empty:
        return patient

    h = test.copy()
    unit = _as_str_lower(h["unit"])

    is_pct  = unit.isin({"%","percent","ngsp","ngsp %"})
    is_ifcc = unit.isin({"mmol/mol","mmol per mol","ifcc"})

    # explicit IFCC conversion
    h.loc[is_ifcc, "value"] = 0.09148 * h.loc[is_ifcc, "value"] + 2.152

    # infer when unit missing/opaque
    missingish = unit.isna() | (unit == "nan") | (unit == "")
    infer_pct  = missingish & h["value"].between(2, 20, inclusive="both")
    infer_ifcc = missingish & (~h["value"].between(2, 20, inclusive="both")) & h["value"].notna()
    h.loc[infer_ifcc, "value"] = 0.09148 * h.loc[infer_ifcc, "value"] + 2.152
    # infer_pct → already percent, no change

    # plausibility in %
    h = h[(h["value"].notna()) & (h["value"] > 2.0) & (h["value"] <= 20.0)].copy()
    if h.empty:
        return patient

    # nearest to indexdate
    h = h[h["patid"].isin(patient["patid"])]
    h = h.merge(patient[["patid","indexdate"]], on="patid", how="left")
    h = h.dropna(subset=["indexdate","obsdate"])
    h["gap"] = (h["indexdate"] - h["obsdate"]).abs().dt.days
    h = h.loc[h.groupby(["patid","indexdate"])["gap"].idxmin()].copy()

    h = h.rename(columns={"obsdate":"hba1c_date","value":"hba1c_perc"})
    out = patient.merge(h[["patid","indexdate","hba1c_date","hba1c_perc"]],
                        on=["patid","indexdate"], how="left")
    return out

# =========================================================
# ======== Apply EXACT functions to your Aurum data =======
# =========================================================

# Lipids (same caps & names as GOLD)
df = get_lab_data(tc,   df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="tot_chol")
df = get_lab_data(hdl,  df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="hdl")
df = get_lab_data(ldl,  df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="ldl")
df = get_lab_data(trig, df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="trigly")

# HbA1c (percent)
df = get_hba1c_data(df, hba1c)

# Save
df.to_csv(output_file, index=False)
print("Saved:", output_file)
