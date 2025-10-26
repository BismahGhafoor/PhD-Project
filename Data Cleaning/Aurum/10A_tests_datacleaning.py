import pandas as pd
import numpy as np
import glob
import os

# --------------------------
# Paths
# --------------------------
chunk_folder  = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/biomarker_tmp_outputs_txt"  # folder with *_chunk_####.txt.gz
output_file   = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/FINAL_Aurum_with_Tests.txt"
enriched_file = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/Enriched_Aurum_with_Biomarkers.txt"

# ---------- Load cohort ----------
# IMPORTANT: enriched_file is a TSV
df = pd.read_csv(enriched_file, sep="\t", dtype=str)
df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce")
df["dod"]       = pd.to_datetime(df["dod"], errors="coerce")
df["patid"]     = df["patid"].astype(str)

# ---------- Chunk loader (supports .txt.gz and .csv.gz) ----------
def load_test_chunks(prefix: str) -> pd.DataFrame:
    # look for both TSV and CSV chunks
    pats = [
        os.path.join(chunk_folder, f"{prefix}_chunk_*.txt.gz"),
        os.path.join(chunk_folder, f"{prefix}_chunk_*.csv.gz"),
    ]
    files = sorted([f for pat in pats for f in glob.glob(pat)])
    if not files:
        return pd.DataFrame(columns=["patid", "obsdate", "value", "unit"])

    parts = []
    for f in files:
        sep = "\t" if f.endswith(".txt.gz") else ","
        x = pd.read_csv(f, dtype=str, sep=sep, compression="gzip")
        for c in ("patid", "obsdate", "value"):
            if c not in x.columns:
                x[c] = pd.NA
        if "unit" not in x.columns:
            x["unit"] = pd.NA
        x = x[["patid", "obsdate", "value", "unit"]]
        parts.append(x)

    out = pd.concat(parts, ignore_index=True)
    out["patid"] = out["patid"].astype(str)
    out["obsdate"] = pd.to_datetime(out["obsdate"], errors="coerce", dayfirst=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["obsdate"].notna()]
    return out

hdl   = load_test_chunks("hdl")
ldl   = load_test_chunks("ldl")
trig  = load_test_chunks("triglycerides")
hba1c = load_test_chunks("hba1c")
tc    = load_test_chunks("tot_chol")

# =========================================================
# Helpers for unit handling (lipids) and HbA1c conversion
# =========================================================
def _as_str_lower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _standardize_lipid_units_like_gold(x: pd.DataFrame, var: str) -> pd.DataFrame:
    if x.empty:
        return x
    y = x.copy()
    unit = _as_str_lower(y["unit"])

    mgdl_syn  = {"mg/dl", "mgdl", "mg per dl", "mg%/dl", "mg%dl"}
    # mmol/L synonyms are not actually needed beyond plausibility filter
    factor = 0.01129 if var == "triglycerides" else 0.02586

    is_mgdl = unit.isin(mgdl_syn)
    y.loc[is_mgdl, "value"] = y.loc[is_mgdl, "value"] * factor

    # missing/unknown unit: big values likely mg/dL
    missingish = (unit == "nan") | (unit == "")
    looks_mgdl = missingish & (y["value"] > 40)
    y.loc[looks_mgdl, "value"] = y.loc[looks_mgdl, "value"] * factor

    # plausible mmol/L range
    y = y[(y["value"].notna()) & (y["value"] >= 0) & (y["value"] <= 20)]
    return y

def get_lab_data(test: pd.DataFrame, patient: pd.DataFrame, enttype_ignored, unit_code_like_96, limit, col):
    if test.empty:
        return patient

    var_name = {"tot_chol": "tot_chol", "hdl": "hdl", "ldl": "ldl", "trigly": "triglycerides"}.get(col, col)
    t = _standardize_lipid_units_like_gold(test, var_name)

    t = t[(t["value"].notna()) & (t["value"] <= limit)].copy()
    if t.empty:
        return patient

    t = t[t["patid"].isin(patient["patid"])]
    t = t.merge(patient[["patid", "indexdate"]], on="patid", how="left")
    t = t.dropna(subset=["indexdate", "obsdate"])
    t["gap"] = (t["indexdate"] - t["obsdate"]).abs().dt.days
    t = t.loc[t.groupby(["patid", "indexdate"])["gap"].idxmin()].copy()

    t = t.rename(columns={"obsdate": f"{col}_date", "value": col})
    out = patient.merge(t[["patid", "indexdate", f"{col}_date", col]], on=["patid", "indexdate"], how="left")
    return out

def get_hba1c_data(patient: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if test.empty:
        return patient
    h = test.copy()
    unit = _as_str_lower(h["unit"])

    is_pct  = unit.isin({"%", "percent", "ngsp", "ngsp %"})
    is_ifcc = unit.isin({"mmol/mol", "mmol per mol", "ifcc"})

    # explicit IFCC conversion (mmol/mol -> %)
    h.loc[is_ifcc, "value"] = 0.09148 * h.loc[is_ifcc, "value"] + 2.152

    # infer if unit missing/opaque
    missingish = (unit == "nan") | (unit == "")
    infer_pct  = missingish & h["value"].between(2, 20, inclusive="both")
    infer_ifcc = missingish & (~h["value"].between(2, 20, inclusive="both")) & h["value"].notna()
    h.loc[infer_ifcc, "value"] = 0.09148 * h.loc[infer_ifcc, "value"] + 2.152
    # infer_pct -> already in %

    # plausibility in %
    h = h[(h["value"].notna()) & (h["value"] > 2.0) & (h["value"] <= 20.0)].copy()
    if h.empty:
        return patient

    h = h[h["patid"].isin(patient["patid"])]
    h = h.merge(patient[["patid", "indexdate"]], on="patid", how="left")
    h = h.dropna(subset=["indexdate", "obsdate"])
    h["gap"] = (h["indexdate"] - h["obsdate"]).abs().dt.days
    h = h.loc[h.groupby(["patid", "indexdate"])["gap"].idxmin()].copy()

    h = h.rename(columns={"obsdate": "hba1c_date", "value": "hba1c_perc"})
    out = patient.merge(h[["patid", "indexdate", "hba1c_date", "hba1c_perc"]], on=["patid", "indexdate"], how="left")
    return out

# =========================================================
# Apply to Aurum data (lipids + HbA1c)
# =========================================================
df = get_lab_data(tc,   df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="tot_chol")
df = get_lab_data(hdl,  df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="hdl")
df = get_lab_data(ldl,  df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="ldl")
df = get_lab_data(trig, df, enttype_ignored=None, unit_code_like_96=96.0, limit=20, col="trigly")
df = get_hba1c_data(df, hba1c)

# Save final TSV
df.to_csv(output_file, sep="\t", index=False)
print("Saved:", output_file)
