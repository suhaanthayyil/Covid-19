# Real COVID-19 Data for Research

This pipeline uses **real, citable datasets** (no synthetic data in the main workflow).

## 1. Download real data (recommended)

From the project root, run:

```bash
python scripts/download_real_data.py
```

- **If you have Kaggle configured** (`pip install kaggle`, and `~/.kaggle/kaggle.json` with your username and API key from https://www.kaggle.com/settings): downloads the **Mexico COVID-19 dataset** (meirnizri/covid19-dataset, 1M+ rows) into `data/raw/`. Best for publication.
- **If Kaggle is not set up**: downloads the **PhysioNet COVID-19 in MS dataset**. Citation: Khan et al. (2024) PhysioNet. https://doi.org/10.13026/77ta-1866

**Security:** Do not commit `kaggle.json` or share your API key. If a key was exposed, create a new one at Kaggle and replace it in `~/.kaggle/kaggle.json`.

## 2. Manual download (optional)

### Mexico COVID-19 (Kaggle) — best for papers

- https://www.kaggle.com/datasets/meirnizri/covid19-dataset  
- Place the CSV (or unzipped CSV) in `data/raw/` as `covid19-dataset.csv` or `mexico_covid.csv`.

Required columns: `CLASIFFICATION_FINAL` (or `CLASIFICATION_FINAL`), `PATIENT_TYPE`, `AGE`, `SEX`, plus comorbidities: `PNEUMONIA`, `DIABETES`, `ASTHMA`, `HIPERTENSION`, `OBESITY`, `CARDIOVASCULAR`, `RENAL_CHRONIC`, `TOBACCO`.

### PhysioNet COVID-19 in MS

- https://physionet.org/content/patient-level-data-covid-ms/  
- Or let `download_real_data.py` fetch it (no login).

## 3. Long COVID definition in the pipeline

- **Mexico data:** Long COVID = confirmed positive (classification 1–3) and `PATIENT_TYPE == 2` (hospitalised / returned to hospital).
- **PhysioNet data:** Long COVID proxy = hospitalisation (outcome level 1 or 2: hospitalised or ICU/ventilation).

Do not commit raw CSV files to git (large files, license). Use `data/raw/.gitkeep` only.
