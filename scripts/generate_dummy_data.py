#!/usr/bin/env python3
"""
Generate a small synthetic COVID-19-style CSV for testing the pipeline
without downloading the full Kaggle dataset. Place output in data/raw/.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "raw", "covid19-dataset.csv")

np.random.seed(42)
n = 50_000  # Small enough to run quickly

# Columns expected by the pipeline (Mexico COVID-19 style)
df = pd.DataFrame({
    "CLASIFFICATION_FINAL": np.random.choice([1, 2, 3], size=n),  # Confirmed positive
    "PATIENT_TYPE": np.random.choice([1, 2], size=n, p=[0.85, 0.15]),  # 15% "Long COVID"
    "AGE": np.clip(np.random.normal(45, 20, n).astype(int), 0, 100),
    "SEX": np.random.choice([1, 2], size=n),
    "PNEUMONIA": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.1, 0.7, 0.05, 0.1, 0.05]),
    "DIABETES": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.15, 0.7, 0.05, 0.05, 0.05]),
    "ASTHMA": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.05, 0.8, 0.05, 0.05, 0.05]),
    "HIPERTENSION": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.2, 0.65, 0.05, 0.05, 0.05]),
    "OBESITY": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.2, 0.65, 0.05, 0.05, 0.05]),
    "CARDIOVASCULAR": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.05, 0.8, 0.05, 0.05, 0.05]),
    "RENAL_CHRONIC": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.03, 0.82, 0.05, 0.05, 0.05]),
    "TOBACCO": np.random.choice([1, 2, 97, 98, 99], size=n, p=[0.1, 0.75, 0.05, 0.05, 0.05]),
})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved synthetic dataset ({n:,} rows) to {OUT_PATH}")
print("Run: python run_pipeline.py")
