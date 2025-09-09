#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one GOLD data cleaning & feature construction pipeline.

Stages:
  A) Code list filtering (keep diabetes type 1/2 codes; add terminology=medcode)
  B) Clinical/Therapy/Test extraction (reads .zip -> .txt inside, chunking + optional code-filter)
  C) Baseline cohort derivation (earliest diabetes clinical record per patid, assign diabetes_type)
  D) Merge demographics & linkages (gender, yob, ethnicity, IMD2019 decile, date of death)
  E) Clinical Smoking medcode extract (from clinical files using Excel code list)
  F) Additional Clinical (enttype) chunked extraction -> smoking, BMI (from weight/height), BP
  G) Test entities assembly (chunked) -> lab panel
  H) Labs extraction (chol, HDL, LDL, triglycerides, HbA1c with unit conversions)
  I) Output final tidy patient-level dataset

Notes:
  • Requires pandas, numpy; some stages reference helper_functions.* (same as your scripts 6 & 8).
  • Paths and switches are grouped in CONFIG. Update to your environment as needed.
  • Designed to be idempotent: will (re)create intermediates if missing, can skip with flags.

Author: consolidated for Bismah (2025-09-09)
"""

# ===============================
# Imports
# ===============================
import os, glob, zipfile, time, platform, warnings, tempfile
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# ===============================
# CONFIG (EDIT ME)
# ===============================
# Base working dirs (Windows vs Linux)
WORKDIR_WIN = r'C:\path\to\your\workdir'
WORKDIR_LIN = '/rfs/LRWE_Proj88/bg205/DataAnalysis'

# CPRD sources (GOLD)
CLIN_ZIPS_GLOB   = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"
THERAPY_GLOB     = "GOLD/Therapy/*.zip"
TEST_GLOB        = "GOLD/Test/*.zip"

# Codelists
DIAB_CODE_DOC    = "/rfs/LRWE_Proj88/Shared/Codes/GOLD_final.txt"  # tab: includes 'medcode' and 'type'
FILTERED_CODES   = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Medcode_filtering/filtered_diabetes_codes.txt"
THERAPY_CODELIST = "final_codelist_gold_therapy.txt"
TEST_CODELIST    = "final_codelist_gold_test.txt"

# Baseline & linkage files
PATIENT_TXT      = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Patient_001.txt"
ETHNICITY_TXT    = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_patient_23_002869_DM.txt"
IMD2019_TXT      = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/patient_2019_imd_23_002869.txt"
DEATH_TXT        = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/death_patient_23_002869_DM.txt"

# Smoking codes (Excel, sheet Smok)
SMOKING_XLSX     = "GOLD_Codes_FZ.xlsx"
SMOK_SHEET       = "Smok"

# HES hospital dx (for smoking ICD flags)
HES_HOSP_TXT     = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_diagnosis_hosp_23_002869_DM.txt"

# Test entities lookups for HbA1c unit names
SUM_UNITS_TXT    = "/rfs/LRWE_Proj88/Shared/Cohort_Definition/Denominator_Linkage_CPRD_Cohort_Data/GOLD_Lookups/GOLD/TXTFILES/SUM.txt"

# Chunking / performance
CHUNK_ROWS       = 20000

# Helper functions (optional module used in original scripts 6 & 8)
USE_HELPERS      = True
try:
    if USE_HELPERS:
        from helper_functions import lcf, ucf, perc
        from helper_functions import save_long_format_data, read_long_format_data
        from helper_functions import remap_eth, nperc_counts, calc_gfr
except Exception as e:
    USE_HELPERS = False
    def save_long_format_data(*args, **kwargs): pass
    def nperc_counts(df, col): 
        vc = df[col].value_counts(dropna=False)
        print(f"[nperc_counts] {col}:\n{vc}\n")

# Switches to run stages
RUN_A_CODES        = True
RUN_B_EXTRACTS     = True
RUN_C_BASELINE     = True
RUN_D_MERGE_DEMOS  = True
RUN_E_CLIN_SMOK    = True
RUN_F_ADDCLN       = True
RUN_G_TEST_ENTS    = True
RUN_H_LABS         = True

# ===============================
# Utilities
# ===============================
def cd():
    """Change to OS-appropriate working directory."""
    path = WORKDIR_WIN if platform.system() == 'Windows' else WORKDIR_LIN
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print(f"[cwd] {os.getcwd()}")

def timer(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def read_zip_txt_first(zip_path):
    """Read first .txt inside a .zip into DataFrame (dtype=str)."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        txts = [n for n in z.namelist() if n.endswith('.txt')]
        if not txts:
            raise FileNotFoundError(f"No .txt inside {zip_path}")
        if len(txts) > 1:
            print(f"[warn] {os.path.basename(zip_path)} has multiple .txt; reading {txts[0]}")
        with z.open(txts[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str)

def safe_read_tsv(path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", **kwargs)

# ===============================
# A) Filter diabetes code list
# ===============================
def stage_A_filter_codes():
    timer("A) Filter diabetes codelist (type in {1,2}) and tag terminology=medcode")
    df_codes = pd.read_csv(DIAB_CODE_DOC, sep="\t", dtype={'medcode': str})
    print(df_codes['type'].value_counts())
    df = df_codes[df_codes['type'].isin([1, 2])].copy()
    print("[after filter]\n", df['type'].value_counts())
    df['terminology'] = 'medcode'
    df = df.rename(columns={'medcode': 'code'})
    out = FILTERED_CODES
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[saved] {out}")

# ===============================
# B) Read & optionally filter Clinical/Therapy/Test
# ===============================
def stage_B_extracts(filter_clin=True, filter_ther=False, filter_test=False, max_rows=np.inf):
    timer("B) Extract Clinical/Therapy/Test")
    # Read code files if filtering
    clin_codes = pd.read_csv(FILTERED_CODES, sep="\t", dtype={'code': str}) if filter_clin else None
    ther_codes = pd.read_csv(THERAPY_CODELIST, sep="\t", dtype={'code': str}) if filter_ther else None
    test_codes = pd.read_csv(TEST_CODELIST, sep="\t", dtype={'code': str}) if filter_test else None

    # Clinical from .zip
    clin_files = sorted(glob.glob(CLIN_ZIPS_GLOB))
    assert clin_files, f"No clinical zips at {CLIN_ZIPS_GLOB}"
    medcodes = clin_codes[clin_codes.terminology == 'medcode']['code'].unique().tolist() if clin_codes is not None else []
    entcodes = clin_codes[clin_codes.terminology == 'enttype']['code'].unique().tolist() if (clin_codes is not None and 'enttype' in clin_codes.terminology.unique()) else []

    tmp, chunk = [], 1
    start = time.perf_counter()
    for i, zp in enumerate(clin_files, 1):
        df = read_zip_txt_first(zp)
        if filter_clin:
            keep = pd.Series(False, index=df.index)
            if 'medcode' in df.columns:
                keep |= df['medcode'].isin(medcodes)
            if 'enttype' in df.columns and len(entcodes):
                keep |= df['enttype'].isin(entcodes)
            df = df[keep]
        tmp.extend(df.to_dict('records'))
        if len(tmp) >= max_rows or i == len(clin_files):
            out_df = pd.DataFrame(tmp)
            out_df.to_csv(f"Cleaned_GOLD_Extract_Clinical_{chunk}.txt",
                          sep="\t", index=False, date_format='%d/%m/%Y')
            print(f"[clinical] saved chunk {chunk} ({len(out_df):,} rows)")
            tmp, chunk = [], chunk+1
    print(f"[clinical] done in {round((time.perf_counter()-start)/60,2)} min")

    # Therapy (pandas can read zipped .txt directly)
    ther_files = sorted(glob.glob(THERAPY_GLOB))
    if ther_files:
        tmp, chunk, start = [], 1, time.perf_counter()
        prodlist = ther_codes['code'].unique().tolist() if ther_codes is not None else []
        for i, fp in enumerate(ther_files, 1):
            df = pd.read_csv(fp, sep="\t", dtype=str)
            if filter_ther and 'prodcode' in df.columns:
                df = df[df['prodcode'].isin(prodlist)]
            tmp.extend(df.to_dict('records'))
            if len(tmp) >= max_rows or i == len(ther_files):
                out_df = pd.DataFrame(tmp)
                out_df.to_csv(f"Cleaned_GOLD_Extract_Therapy_{chunk}.txt",
                              sep="\t", index=False, date_format='%d/%m/%Y')
                print(f"[therapy] saved chunk {chunk} ({len(out_df):,} rows)")
                tmp, chunk = [], chunk+1
        print(f"[therapy] done in {round((time.perf_counter()-start)/60,2)} min")
    else:
        print("[therapy] no files")

    # Test
    test_files = sorted(glob.glob(TEST_GLOB))
    if test_files:
        tmp, chunk, start = [], 1, time.perf_counter()
        entlist = test_codes['code'].unique().tolist() if test_codes is not None else []
        for i, fp in enumerate(test_files, 1):
            df = pd.read_csv(fp, sep="\t", dtype=str)
            if filter_test and 'enttype' in df.columns:
                df = df[df['enttype'].isin(entlist)]
            tmp.extend(df.to_dict('records'))
            if len(tmp) >= max_rows or i == len(test_files):
                out_df = pd.DataFrame(tmp)
                out_df.to_csv(f"Cleaned_GOLD_Extract_Test_{chunk}.txt",
                              sep="\t", index=False, date_format='%d/%m/%Y')
                print(f"[test] saved chunk {chunk} ({len(out_df):,} rows)")
                tmp, chunk = [], chunk+1
        print(f"[test] done in {round((time.perf_counter()-start)/60,2)} min")
    else:
        print("[test] no files")

# ===============================
# C) Baseline from clinical chunks + diabetes type
# ===============================
def stage_C_build_baseline(clin_chunk_glob, filtered_codes_path, outdir="Baseline_dataframe/Cleaned_baselinedata"):
    timer("C) Build baseline (earliest diabetes clinical record per patid) & assign type")
    os.makedirs(outdir, exist_ok=True)

    chunk_files = sorted(glob.glob(clin_chunk_glob))
    assert chunk_files, f"No clinical chunk files at {clin_chunk_glob}"

    def _proc(f):
        df = pd.read_csv(f, sep="\t", dtype=str)
        keep_cols = [c for c in ["patid","eventdate","medcode"] if c in df.columns]
        df = df[keep_cols].copy()
        df["eventdate"] = pd.to_datetime(df["eventdate"], errors='coerce', dayfirst=True)
        df = df.dropna(subset=["patid"])
        df = df.sort_values(["patid","eventdate"]).drop_duplicates("patid", keep="first")
        return df

    baseline = pd.concat([_proc(f) for f in chunk_files], ignore_index=True)
    codes = pd.read_csv(filtered_codes_path, sep="\t", dtype=str).rename(columns={"code":"medcode"})
    m = codes.set_index("medcode")["type"].to_dict()
    baseline["diabetes_type"] = baseline["medcode"].map(m)
    baseline.to_csv(os.path.join(outdir, "baseline_ungrouped_df_WithNA.txt"), sep="\t", index=False)
    # grouped outputs
    t1 = baseline[baseline["diabetes_type"]=="1"]
    t2 = baseline[baseline["diabetes_type"]=="2"]
    t1.to_csv(os.path.join(outdir,"baseline_Type_1_Diabetes_WithNA.txt"), sep="\t", index=False)
    t2.to_csv(os.path.join(outdir,"baseline_Type_2_Diabetes_WithNA.txt"), sep="\t", index=False)
    grp = pd.concat([t1,t2], ignore_index=True)
    grp.to_csv(os.path.join(outdir,"baseline_grouped_df_WithNA.txt"), sep="\t", index=False)
    print(f"[baseline] n T1={t1['patid'].nunique()}  n T2={t2['patid'].nunique()}")

# ===============================
# D) Merge demographics & linkage
# ===============================
def stage_D_merge_demographics(outdir="Baseline_dataframe/Cleaned_baselinedata"):
    timer("D) Merge demographics & linkage (gender,yob,ethnicity,IMD2019,dod)")
    base_path = os.path.join(outdir, "baseline_grouped_df_WithNA.txt")
    df = pd.read_csv(base_path, sep="\t", dtype=str, parse_dates=["eventdate"], dayfirst=True)
    patient = pd.read_csv(PATIENT_TXT, sep="\t", dtype=str)
    eth = pd.read_csv(ETHNICITY_TXT, sep="\t", dtype=str)
    imd = pd.read_csv(IMD2019_TXT, sep="\t", dtype=str)
    dod = pd.read_csv(DEATH_TXT, sep="\t", dtype=str, low_memory=False)

    dfm = (df.merge(patient[['patid','gender','yob']], on='patid', how='left')
             .merge(eth[['patid','gen_ethnicity']], on='patid', how='left')
             .merge(imd[['patid','e2019_imd_10']], on='patid', how='left')
             .merge(dod[['patid','dod']], on='patid', how='left'))
    out = os.path.join(outdir, "baseline_with_all_features.txt")
    dfm.to_csv(out, sep="\t", index=False)
    print(f"[merged] {out}")

# ===============================
# E) Clinical smoking medcodes -> gz
# ===============================
def stage_E_clinical_smoking_extract(output_gz="Clinical_SmokingStatus_all.txt.gz"):
    timer("E) Extract clinical smoking rows using Excel medcodes (Smok)")
    # clinical raw TXT (not zips) expected here:
    clin_txt_glob = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.txt"
    files = sorted(glob.glob(clin_txt_glob))
    assert files, f"No clinical TXT files at {clin_txt_glob}"

    # smoking medcodes
    smok_df = pd.read_excel(SMOKING_XLSX, sheet_name=SMOK_SHEET, dtype=str)
    medcodes = set(smok_df['medcode'].dropna().astype(str))

    tmp = []
    start = time.perf_counter()
    for i, fp in enumerate(files, 1):
        df = pd.read_csv(fp, sep="\t", dtype=str)
        df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["eventdate"])
        keep = df['medcode'].isin(medcodes) if 'medcode' in df.columns else pd.Series(False, index=df.index)
        df = df.loc[keep, ["patid","eventdate","medcode"]]
        tmp.extend(df.to_dict('records'))
        print(f"[clin_smok] file {i}/{len(files)} kept {len(df):,} rows")

    out = pd.DataFrame(tmp)
    out.to_csv(output_gz, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    print(f"[clin_smok] saved {output_gz}  rows={len(out):,}  elapsed={round((time.perf_counter()-start)/60,2)}m")

# ===============================
# F) Additional Clinical -> smoking/BMI/BP
# ===============================
def stage_F_additional_clinicals(out_patient_csv="Cleaned_Patient_Smoking_Data.csv"):
    timer("F) Process Additional Clinical: smoking (enttype 4), BP (1), weight(13)/height(14)")
    # Load baseline with demographics
    patient = pd.read_csv(
        "Baseline_dataframe/Cleaned_baselinedata/baseline_with_all_features.txt",
        sep="\t", dtype=str, parse_dates=["eventdate","dod"], dayfirst=True)
    # Define indexdate = baseline eventdate
    patient = patient.rename(columns={"eventdate":"indexdate"})
    # Clinical smoking gz
    clinical_smok = pd.read_csv("Clinical_SmokingStatus_all.txt.gz", sep="\t",
                                compression="gzip", dtype=str, parse_dates=["eventdate"], dayfirst=True)
    # Additional clinical files (TXT)
    add_glob = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Additional_*.txt"
    add_files = sorted(glob.glob(add_glob))
    assert add_files, f"No Additional clinical files at {add_glob}"

    # Build subset temp files by chunk
    temp_bp, temp_smok, temp_wt, temp_ht = "temp_bp.csv","temp_smoking.csv","temp_weight.csv","temp_height.csv"
    for p in [temp_bp,temp_smok,temp_wt,temp_ht]:
        if os.path.exists(p): os.remove(p)

    for f in add_files:
        for ch in pd.read_csv(f, sep="\t", dtype=str, chunksize=CHUNK_ROWS):
            for ent, tmpf in [("1",temp_bp), ("4",temp_smok), ("13",temp_wt), ("14",temp_ht)]:
                sub = ch[ch["enttype"]==ent]
                if not sub.empty:
                    sub.to_csv(tmpf, mode='a', header=not os.path.exists(tmpf), index=False)

    # Load subsets
    def _load(path): 
        return pd.read_csv(path, dtype=str) if os.path.exists(path) else pd.DataFrame()
    bp_data     = _load(temp_bp)
    smoking_add = _load(temp_smok)
    weight_data = _load(temp_wt)
    height_data = _load(temp_ht)

    # Ensure dates
    def _ensure_dates(df, idx_name="indexdate"):
        if df.empty: return df
        if idx_name not in df.columns:
            df = df.merge(patient[['patid', idx_name]], on='patid', how='left')
        for c in ["eventdate", idx_name]:
            if c not in df.columns: df[c] = df[idx_name]
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        return df

    patient['patid'] = patient['patid'].astype(str)
    patient['indexdate'] = pd.to_datetime(patient['indexdate'], errors="coerce", dayfirst=True)

    bp_data     = _ensure_dates(bp_data)
    smoking_add = _ensure_dates(smoking_add)
    weight_data = _ensure_dates(weight_data)
    height_data = _ensure_dates(height_data)

    # HES hospital smoking signals
    hes = pd.read_csv(HES_HOSP_TXT, sep="\t", dtype=str)
    hes['patid'] = hes['patid'].astype(str)
    hes["admidate"] = pd.to_datetime(hes["admidate"], errors="coerce", dayfirst=True)
    hes = hes.merge(patient[['patid','indexdate']], on='patid', how='left')

    # --- Functions mirroring your Script 6 ---
    def get_smoking(patient):
        # additional clinical (data1 in {1,2,3}) at enttype=4
        if not smoking_add.empty and "data1" in smoking_add.columns:
            sm = smoking_add[['patid','eventdate','indexdate','data1']].copy()
            sm = sm[sm['data1'].isin(["1","2","3"]) & sm['eventdate'].notna()]
            sm['smok_add'] = sm['data1'].replace({"1":"Yes","2":"No","3":"Ex"})
            sm = sm.sort_values(['patid','eventdate']).drop_duplicates(['patid','eventdate'], keep='last')
        else:
            sm = pd.DataFrame(columns=['patid','eventdate','indexdate','smok_add'])

        # clinical medcodes
        smok_codes = pd.read_excel(SMOKING_XLSX, sheet_name="Smok")
        smok_cprd = smok_codes[smok_codes.source=='cprd']
        groups = {
            'current smoker':'Yes',
            'never smoker':'No',
            'ex smoker':'Ex'
        }
        cl = clinical_smok[['patid','eventdate','medcode']].copy()
        cl['smok_clref'] = np.nan
        for key, lab in groups.items():
            medlist = set(smok_cprd[smok_cprd['type']==key]['medcode'].astype(str))
            cl.loc[cl['medcode'].astype(str).isin(medlist),'smok_clref'] = lab
        cl = cl.dropna(subset=['smok_clref'])
        cl = cl.sort_values(['patid','eventdate']).drop_duplicates(['patid','eventdate'], keep='last')
        cl['indexdate'] = cl['patid'].map(patient.set_index('patid')['indexdate'])

        # HES ICD smoking
        smok_hes_codes = set(smok_codes[smok_codes.source=='hes']['medcode'].astype(str))
        if 'ICD' in hes.columns and not pd.isna(list(smok_hes_codes)[:1]).all():
            hes_sm = hes[hes["ICD"].astype(str).str.contains("|".join(smok_hes_codes), na=False)]
        else:
            hes_sm = hes.iloc[0:0].copy()
        hes_sm = hes_sm.rename(columns={'admidate':'eventdate'})
        hes_sm['smok_hes'] = 'Yes'
        hes_sm = hes_sm[['patid','eventdate','indexdate','smok_hes']]

        # merge preference: clinical -> HES -> additional
        df = cl.merge(sm, how='outer', on=['patid','eventdate','indexdate'])
        df = df.merge(hes_sm, how='outer', on=['patid','eventdate','indexdate'])
        df['smoking_status'] = df['smok_clref']
        df.loc[df['smoking_status'].isna() & df['smok_hes'].notna(),'smoking_status'] = df['smok_hes']
        df.loc[df['smoking_status'].isna() & df['smok_add'].notna(),'smoking_status'] = df['smok_add']

        if USE_HELPERS: nperc_counts(df, 'smoking_status')
        df['smok_time_gap'] = (df['indexdate'] - df['eventdate']).abs().dt.days
        df = df.loc[df.groupby(['patid','indexdate'])['smok_time_gap'].idxmin()]
        df = df[['patid','indexdate','eventdate','smoking_status']].rename(columns={'eventdate':'smoking_date'})
        return patient.merge(df, on=['patid','indexdate'], how='left')

    def get_bmi(patient):
        wh = pd.concat([weight_data, height_data], ignore_index=True)
        wt = wh[wh['enttype']=="13"][['patid','eventdate','indexdate','data1','data3']].rename(
            columns={'data1':'weight_kg','data3':'bmi_recorded'})
        ht = wh[wh['enttype']=="14"][['patid','eventdate','indexdate','data1']].rename(
            columns={'data1':'height_m'})
        bmi = wt.merge(ht, how='outer', on=['patid','eventdate','indexdate'])
        for c in ['weight_kg','height_m','bmi_recorded']:
            bmi[c] = pd.to_numeric(bmi[c], errors='coerce')
        bmi['height_m'] = bmi.groupby(['patid','indexdate'])['height_m'].ffill()
        bmi['bmi_calc'] = bmi['weight_kg']/(bmi['height_m']*bmi['height_m'])
        bmi['bmi'] = bmi['bmi_recorded']
        mask_replace = (bmi['bmi'].isna() | (bmi['bmi']<=0) | (bmi['bmi']==np.inf)) & (bmi['bmi_calc']<np.inf)
        bmi.loc[mask_replace,'bmi'] = bmi['bmi_calc']
        bmi = bmi[(bmi['bmi']>=10) & (bmi['bmi']<=70)]
        bmi = (bmi.drop_duplicates(['patid','indexdate','eventdate','bmi'], keep='last')
                  .groupby(['patid','indexdate','eventdate']).mean().reset_index())
        save_long_format_data(bmi, False, 'bmi')
        bmi['bmi_time_gap'] = (bmi['indexdate'] - bmi['eventdate']).abs().dt.days
        bmi = bmi.loc[bmi.groupby(['patid','indexdate'])['bmi_time_gap'].idxmin()]
        bmi = bmi[['patid','indexdate','eventdate','bmi']].rename(columns={'eventdate':'bmi_date'})
        return patient.merge(bmi, on=['patid','indexdate'], how='left')

    def get_bp(patient):
        if bp_data.empty:
            return patient
        bp = bp_data[['patid','eventdate','indexdate','data1','data2']].rename(columns={'data1':'diastolic','data2':'systolic'})
        bp = bp.dropna(subset=['eventdate','diastolic','systolic'])
        bp[['diastolic','systolic']] = bp[['diastolic','systolic']].apply(pd.to_numeric, errors='coerce')
        bp = bp[(bp['systolic'].between(20,300)) & (bp['diastolic'].between(5,200))]
        bp = (bp.drop_duplicates().groupby(['patid','indexdate','eventdate']).mean().reset_index())
        save_long_format_data(bp, False, 'bp')
        bp['bp_time_gap'] = (bp['indexdate'] - bp['eventdate']).abs().dt.days
        bp = bp.loc[bp.groupby(['patid','indexdate'])['bp_time_gap'].idxmin()]
        bp = bp[['patid','indexdate','eventdate','systolic','diastolic']].rename(columns={'eventdate':'bp_date'})
        return patient.merge(bp, on=['patid','indexdate'], how='left')

    # run
    patient = get_smoking(patient)
    patient = get_bmi(patient)
    patient = get_bp(patient)
    patient.to_csv(out_patient_csv, index=False)
    print(f"[add_clin] saved {out_patient_csv} (n={len(patient):,})")

# ===============================
# G) Build Test_entities_all.txt.gz
# ===============================
def stage_G_build_test_entities(out_gz="Test_entities_all.txt.gz"):
    timer("G) Create Test_entities_all.txt.gz (chunked)")
    raw_glob = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.txt"
    files = sorted(glob.glob(raw_glob))
    assert files, f"No test files at {raw_glob}"
    if os.path.exists(out_gz): os.remove(out_gz)

    start = time.perf_counter()
    for i, fp in enumerate(files, 1):
        for chunk in pd.read_csv(fp, sep="\t", dtype=str, chunksize=CHUNK_ROWS):
            chunk["eventdate"] = pd.to_datetime(chunk["eventdate"], errors="coerce", dayfirst=True)
            chunk = chunk.dropna(subset=["eventdate"])
            idx = (chunk.groupby("patid", as_index=False)["eventdate"].min()
                         .rename(columns={"eventdate":"indexdate"}))
            chunk = chunk.merge(idx, on="patid", how="left")
            cols = [c for c in ["patid","eventdate","indexdate","enttype","data1","data2"] if c in chunk.columns]
            chunk = chunk[cols].rename(columns={"data1":"value", "data2":"unit"})
            chunk.to_csv(out_gz, mode='a', header=not os.path.exists(out_gz),
                         index=False, sep="\t", compression="gzip", date_format='%d/%m/%Y')
        print(f"[test_entities] processed {i}/{len(files)}")
    print(f"[test_entities] saved {out_gz} in {round((time.perf_counter()-start)/60,2)} min")

# ===============================
# H) Extract labs (chol, HDL, LDL, TG, HbA1c)
# ===============================
def stage_H_extract_labs(in_patient_csv="Cleaned_Patient_Smoking_Data.csv",
                         in_test_gz="Test_entities_all.txt.gz",
                         out_csv="extracted_lab_data.csv"):
    timer("H) Extract labs (chol/HDL/LDL/TG/HbA1c)")
    patient = pd.read_csv(in_patient_csv, dtype=str, parse_dates=['indexdate','dod','smoking_date','bmi_date','bp_date'], dayfirst=True)
    patient_ids = set(patient['patid'].astype(str).unique())

    # Read test in chunks; keep only patient IDs
    usecols = ["patid","enttype","value","unit","eventdate"]
    test_chunks = []
    for i, ch in enumerate(pd.read_csv(in_test_gz, sep="\t", compression="gzip",
                                       usecols=usecols, dtype=str,
                                       parse_dates=['eventdate'], dayfirst=True, chunksize=100000), 1):
        sub = ch[ch['patid'].isin(patient_ids)]
        test_chunks.append(sub)
        print(f"[labs] chunk {i} -> kept {len(sub):,}/{len(ch):,}")
    test = pd.concat(test_chunks, ignore_index=True)
    test['enttype'] = test['enttype'].astype(str)
    test["indexdate"] = test["patid"].map(patient.set_index("patid")["indexdate"])

    def get_lab(enttype, unit, limit, col):
        df = test[test["enttype"]==str(enttype)].copy()
        # unit/value numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["unit"]  = pd.to_numeric(df["unit"], errors="coerce")
        df = df[(df["unit"]==unit) & (df["value"]<=limit)]
        df[col] = df["value"]
        df = df[["patid","eventdate","indexdate",col]].sort_values(["patid","eventdate"])
        save_long_format_data(df, False, col)
        df["time_gap"] = (df["indexdate"] - df["eventdate"]).abs().dt.days
        df = df.loc[df.groupby("patid")["time_gap"].idxmin()]
        df = df.rename(columns={"eventdate":f"{col}_date"})
        return df[["patid","indexdate",f"{col}_date",col]]

    # convert and merge standard lipids (unit code 96.0, limit 20)
    for ent, col in [("163","tot_chol"),("175","hdl"),("177","ldl"),("202","trigly")]:
        lab = get_lab(enttype=ent, unit=96.0, limit=20, col=col)
        patient = patient.merge(lab, on=["patid","indexdate"], how="left")

    # HbA1c (enttype 275) with unit conversions
    hba1c = test[test["enttype"]=="275"].copy()
    # Map unit code -> unit name (optional)
    try:
        units = pd.read_csv(SUM_UNITS_TXT, sep="\t")
        hba1c["unit_name"] = hba1c["unit"].map(units.set_index("Code")["Specimen Unit of Measure"])
    except Exception:
        hba1c["unit_name"] = np.nan

    # numeric
    hba1c["value"] = pd.to_numeric(hba1c["value"], errors="coerce")
    hba1c["unit"]  = pd.to_numeric(hba1c["unit"], errors="coerce")

    # Convert to % following your mapping
    hba1c["hba1c_perc"] = np.nan
    hba1c.loc[hba1c["unit"].isin([1,215]), "hba1c_perc"] = hba1c["value"]                                # already %
    hba1c.loc[hba1c["unit"].isin([97,156,205,187]), "hba1c_perc"] = hba1c["value"]*0.0915 + 2.15        # mmol/mol -> %
    hba1c.loc[hba1c["unit"].isin([96]), "hba1c_perc"] = hba1c["value"]*0.6277 + 1.627                   # alt mapping

    # clean & pick closest to indexdate
    hba1c = hba1c[(hba1c["hba1c_perc"]>2.0) & (hba1c["hba1c_perc"]<=20)]
    hba1c = (hba1c[['patid','eventdate','indexdate','hba1c_perc']]
                .drop_duplicates()
                .groupby(['patid','indexdate','eventdate']).mean().reset_index())
    save_long_format_data(hba1c, False, 'hba1c')
    hba1c['gap'] = (hba1c['eventdate'] - hba1c['indexdate']).dt.days
    hba1c = hba1c.loc[hba1c.groupby(['patid','indexdate'])['gap'].apply(lambda s: s.abs().idxmin())]
    hba1c = hba1c.rename(columns={'eventdate':'hba1c_date'})
    patient = patient.merge(hba1c[['patid','indexdate','hba1c_date','hba1c_perc']],
                            on=['patid','indexdate'], how='left')

    patient.to_csv(out_csv, index=False)
    print(f"[labs] saved {out_csv} (n={len(patient):,})")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    cd()

    if RUN_A_CODES:
        stage_A_filter_codes()

    if RUN_B_EXTRACTS:
        stage_B_extracts(filter_clin=True, filter_ther=False, filter_test=False, max_rows=np.inf)

    if RUN_C_BASELINE:
        stage_C_build_baseline(
            clin_chunk_glob="Cleaned_GOLD_Extract_Clinical_*.txt",
            filtered_codes_path=FILTERED_CODES
        )

    if RUN_D_MERGE_DEMOS:
        stage_D_merge_demographics()

    if RUN_E_CLIN_SMOK:
        stage_E_clinical_smoking_extract()

    if RUN_F_ADDCLN:
        stage_F_additional_clinicals()

    if RUN_G_TEST_ENTS:
        stage_G_build_test_entities()

    if RUN_H_LABS:
        stage_H_extract_labs()
