#!/usr/bin/env python3
"""
Download real COVID-19 datasets for the Long COVID ML pipeline (research use).

1. Tries Kaggle: meirnizri/covid19-dataset (Mexico, 1M+ rows).
   Requires: pip install kaggle and ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY).
2. Fallback: PhysioNet COVID-19 in MS dataset (real, citable, no login).
   Citation: Khan et al. (2024) PhysioNet. https://doi.org/10.13026/77ta-1866
"""

import os
import sys
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PHYSIONET_URL = "https://physionet.org/files/patient-level-data-covid-ms/1.0.1/GDSI_OpenDataset_Final.csv"
PHYSIONET_OUT = "physionet_covid_ms.csv"
KAGGLE_DATASET = "meirnizri/covid19-dataset"
KAGGLE_OUT = "covid19-dataset.zip"


def download_physionet() -> str:
    """Download PhysioNet COVID-19 in MS dataset (real patient-level data)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, PHYSIONET_OUT)
    print(f"Downloading PhysioNet COVID-19 in MS dataset...")
    try:
        urllib.request.urlretrieve(PHYSIONET_URL, path)
        print(f"Saved to {path}")
        return path
    except Exception as e:
        print(f"PhysioNet download failed: {e}")
        raise


def download_kaggle() -> bool:
    """Try to download Kaggle Mexico COVID-19 dataset. Returns True if successful."""
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, KAGGLE_OUT)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", RAW_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
        # Kaggle downloads a zip; unzip if present
        import zipfile
        zip_path = os.path.join(RAW_DIR, KAGGLE_OUT)
        if os.path.isfile(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(RAW_DIR)
            print(f"Downloaded and extracted to {RAW_DIR}")
            return True
        return True
    except FileNotFoundError:
        print("Kaggle CLI not found. Install: pip install kaggle")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e.stderr or e}")
        return False


def main():
    print("=== Real COVID-19 data download ===\n")
    # Try Kaggle first (larger, Mexico schema)
    if download_kaggle():
        # Check we have a CSV (name may vary inside zip)
        for f in os.listdir(RAW_DIR):
            if f.endswith(".csv") and "covid" in f.lower():
                print(f"Use CSV: data/raw/{f}")
                return
        print("Kaggle zip extracted; check data/raw/ for CSV filename.")
        return
    # Fallback: PhysioNet (no login, real research data)
    print("Using PhysioNet COVID-19 in MS dataset (real, citable).\n")
    download_physionet()
    print("Run pipeline: python run_pipeline.py (preprocessing will auto-detect PhysioNet format).")


if __name__ == "__main__":
    main()
