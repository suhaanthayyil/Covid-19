#!/usr/bin/env python3
"""
Full Long COVID prediction pipeline: load data, preprocess, train 4 models,
evaluate, plot all figures, run SHAP, save metrics. Run from project root:
  python run_pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Project root and figures/results dirs
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_RAW = os.path.join(ROOT, "data", "raw")

FIGURE_DPI = 300
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = FIGURE_DPI

# ---------------------------------------------------------------------------
# Imports from our modules
# ---------------------------------------------------------------------------
from preprocess import (
    load_raw_data,
    detect_data_source,
    find_classification_column,
    create_target_and_filter,
    create_target_and_filter_physionet,
    prepare_features_and_target,
    prepare_features_and_target_physionet,
    train_test_split_and_scale,
    FEATURE_COLS,
)
from train import train_all_models, MODEL_CONFIGS
from evaluate import (
    evaluate_all_models,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_cv_accuracy_box,
)
from explain import run_shap_for_best_model


def fig1_class_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart: Long COVID (1) vs No Long COVID (0) in original data before SMOTE."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    counts = df["LONG_COVID"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=["#2ecc71", "#e74c3c"])
    ax.set_xlabel("Long COVID")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Long COVID vs. No Long COVID — Class Distribution")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (No)", "1 (Yes)"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.01 * counts.values.max(), str(v), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def fig2_correlation_heatmap(X: pd.DataFrame, y: pd.Series, save_path: str) -> None:
    """Seaborn Pearson correlation heatmap (features + target), blue-red colormap."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df = X.copy()
    df["LONG_COVID"] = y.values
    corr = df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    print("=== Long COVID ML Pipeline ===\n")

    # --- Load data (with try/except) ---
    try:
        raw = load_raw_data(data_dir=DATA_RAW)
    except FileNotFoundError as e:
        print(str(e))
        print("Run: python scripts/download_real_data.py (see data/README.md). Exiting.")
        sys.exit(1)

    source = detect_data_source(raw)
    print(f"Detected dataset: {source}")

    if source == "mexico":
        class_col = find_classification_column(raw)
        df_filtered = create_target_and_filter(raw, class_col)
        feature_cols = [c for c in FEATURE_COLS if c in df_filtered.columns]
        X, y = prepare_features_and_target(
            df_filtered, feature_cols=feature_cols, drop_pct_missing=0.30, random_state=42
        )
    else:
        df_filtered = create_target_and_filter_physionet(raw)
        X, y = prepare_features_and_target_physionet(df_filtered, drop_pct_missing=0.50)

    # Optional: cap size for faster run (real data, smaller sample — set MAX_TRAIN_SAMPLES for full run)
    max_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
    if max_samples > 0 and len(X) > max_samples:
        from sklearn.model_selection import train_test_split as tts
        X, _, y, _ = tts(X, y, train_size=max_samples, stratify=y, random_state=42)
        print(f"Using stratified sample of {len(X):,} for training (MAX_TRAIN_SAMPLES={max_samples}).")

    # --- FIGURE 1: Class distribution (before SMOTE) ---
    fig1_class_distribution(df_filtered, os.path.join(FIGURES_DIR, "fig1_class_distribution.png"))

    # --- Preprocessing: split, scale, SMOTE ---
    print("\nPreprocessing...")
    fig2_correlation_heatmap(X, y, os.path.join(FIGURES_DIR, "fig2_correlation_heatmap.png"))

    X_train, X_test, y_train, y_test, scaler = train_test_split_and_scale(
        X, y, test_size=0.2, random_state=42, use_smote=True
    )
    feature_names = list(X.columns)

    # --- Train all 4 models (2-fold for large data to keep run time reasonable; 3-fold if tiny; else 5-fold) ---
    n_pos = int(y_train.sum())
    n_train = len(y_train)
    if n_train > 50_000:
        cv_folds = 2
    elif n_train < 200 or n_pos < 5:
        cv_folds = 3
    else:
        cv_folds = 5
    print(f"\n--- Model training ({cv_folds}-fold CV) ---")
    models, cv_results = train_all_models(X_train, y_train, cv=cv_folds)

    # --- Evaluate: metrics table ---
    metrics_df = evaluate_all_models(models, X_test, y_test, cv_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "metrics_table.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    # --- FIGURE 3: Model comparison bar chart ---
    plot_model_comparison(metrics_df, os.path.join(FIGURES_DIR, "fig3_model_comparison.png"))

    # --- FIGURE 4: ROC curves ---
    plot_roc_curves(models, X_test, y_test.values, os.path.join(FIGURES_DIR, "fig4_roc_curves.png"))

    # --- Best model (highest ROC-AUC; if all NaN use F1_Score), FIGURE 5: Confusion matrix ---
    roc_ok = metrics_df["ROC_AUC"].notna()
    if roc_ok.any():
        best_idx = metrics_df["ROC_AUC"].idxmax()
    else:
        best_idx = metrics_df["F1_Score"].idxmax()
    best_row = metrics_df.loc[best_idx]
    best_name = best_row["Model"]
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(
        y_test.values, y_pred_best,
        save_path=os.path.join(FIGURES_DIR, "fig5_confusion_matrix.png"),
        title="Confusion Matrix — Best Model",
    )

    # --- FIGURE 6: CV accuracy box plot ---
    # Rebuild models for CV box (same configs, fit again for consistency)
    from sklearn.model_selection import cross_val_score
    plot_cv_accuracy_box(models, X_train, y_train, cv=cv_folds, save_path=os.path.join(FIGURES_DIR, "fig6_cv_accuracy.png"))

    # --- SHAP: best model → FIGURE 7 (beeswarm), FIGURE 8 (bar) ---
    print("\n--- SHAP explainability (best model) ---")
    try:
        shap_values, top_feature = run_shap_for_best_model(
            best_model, best_name, X_train, X_test, figures_dir=FIGURES_DIR
        )
    except Exception as e:
        print(f"SHAP failed (optional): {e}")
        top_feature = "N/A"

    # --- Final summary to console ---
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("\n--- Best model ---")
    print(f"  Model: {best_name}")
    print(f"  ROC-AUC: {best_row['ROC_AUC']:.4f}")
    print(f"  Highest mean |SHAP| feature: {top_feature}")
    print("=" * 60)


if __name__ == "__main__":
    main()
