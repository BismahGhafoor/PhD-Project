# Predicting Long-Term Diabetes Complications (ML vs Traditional Models)

This repository contains the code and documentation for my PhD project focused on predicting **long-term diabetes complications** from UK routinely collected health data. The project builds harmonised analysis datasets and develops **machine-learning models** side-by-side with **traditional statistical models** to provide a transparent, fair, and clinically useful comparison.

## Project Overview

Accurately identifying people at risk of diabetes complications (e.g., CVD events, CKD progression, mortality) is critical for timely interventions and better outcomes. This project leverages **CPRD primary care data** linked to **HES** (hospital) and **ONS** (mortality) to engineer high-quality features and evaluate competing modelling approaches.  
We create reproducible cleaning pipelines for **CPRD Aurum** and **CPRD GOLD**, derive risk-factor trajectories (e.g., HbA1c, BP, BMI, lipids, smoking), and compare modern ML methods (e.g., XGBoost) against traditional approaches (e.g., logistic/Cox, Fine–Gray competing risks). Model performance is assessed with discrimination, calibration, decision-curve analysis, and subgroup analyses.

> ⚠️ **Data governance:** CPRD/HES/ONS data are **restricted**. This repository contains **code only**—no patient-level data or keys.

### Objectives
1. Build harmonised, analysis-ready cohorts from **CPRD Aurum** and **CPRD GOLD** with aligned variable names/types.  
2. Engineer robust features for key risk factors (smoking, BMI, BP, HbA1c, lipids, biomarkers) and baseline covariates.  
3. Train and evaluate predictive models:
   - **ML:** gradient boosting/XGBoost, random forests (and extensions as needed).
   - **Traditional:** logistic regression, **Cox**, and **Fine–Gray** competing-risks models.
4. Provide transparent evaluation: AUC/C-index, calibration (calibration curves/ECE), decision-curve analysis, and subgroup/fairness checks (e.g., by age, sex, ethnicity, deprivation).
5. Deliver fully reproducible pipelines (chunked processing + optional SLURM) and clear documentation for reuse.

### Tools and Technologies
- **Python** (data cleaning/feature engineering; pandas/pyarrow/numpy, tqdm, pyyaml)
- **R** (survival/competing-risks analysis; `survival`, `cmprsk`, `riskRegression`, `timeROC`)
- **Machine learning:** scikit-learn, XGBoost
- **Bash & SLURM** for HPC batch processing
- **Conda/uv** for environments; **Git/GitHub** for version control

## Method
1. **Data Access & Governance** — obtain CPRD with HES/ONS linkage under approved protocols.  
2. **Data Pre-processing** — extract/clean Aurum & GOLD; build harmonised baseline datasets.  
3. **Feature Engineering** — derive smoking status, BMI, BP, HbA1c, lipids, and other biomarkers (chunked at scale).  
4. **Modelling** — train ML and traditional models (including competing-risks where appropriate).  
5. **Evaluation** — discrimination, calibration, decision-curve analysis, sensitivity and subgroup analyses.  
6. **Reporting** — reproducible tables/figures and documentation.

## Getting Started

### Prerequisites
- Linux/Mac (or WSL)  
- **Python 3.11+** and (optionally) **R 4.3+**  
- Access to secure storage for CPRD/HES/ONS (not included here)  
- Optional **HPC/SLURM** for large-scale runs

1. Clone the repository:
```bash
git clone https://github.com/BismahGhafoor/PhD-Project.git
cd PhD-Project

## Acknowledgements
Thanks to the NIHR ARC and Leicester Diabetes Research Centre (in particular, Dr. Sharmin Shabnam, Dr. Francesco Zaccardi and Prof. Kamlesh Khunit) for funding and supporting this research.

## Contact
For any questions or inquiries, please contact Bismah Ghafoor at bg205@leicester.ac.uk.

