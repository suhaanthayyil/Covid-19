"""
SHAP explainability: beeswarm summary, bar plot (mean |SHAP|), and waterfall for one sample.
Uses TreeExplainer for RF/XGBoost, LinearExplainer for Logistic Regression,
KernelExplainer (with background sample) for SVM.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

FIGURE_DPI = 300


def get_explainer(model, X_background: pd.DataFrame, model_name: str):
    """
    Return appropriate SHAP explainer for the model type.
    TreeExplainer for RF/XGBoost, LinearExplainer for LR, KernelExplainer for SVM.
    """
    if "XGBoost" in model_name or "Random Forest" in model_name:
        return shap.TreeExplainer(model, X_background)
    if "Logistic Regression" in model_name:
        return shap.LinearExplainer(model, X_background)
    # SVM: use KernelExplainer with a small background to avoid slow runtime
    if "SVM" in model_name:
        bg = X_background.sample(min(100, len(X_background)), random_state=42)
        return shap.KernelExplainer(model.predict_proba, bg)
    # Fallback
    bg = X_background.sample(min(100, len(X_background)), random_state=42)
    return shap.KernelExplainer(model.predict_proba, bg)


def get_shap_values(explainer, X: pd.DataFrame, model_name: str):
    """Compute SHAP values; for binary classification return values for class 1."""
    if "SVM" in model_name or "Kernel" in str(type(explainer)):
        # KernelExplainer returns list for multi-output
        sh = explainer.shap_values(X, nsamples=100)
        if isinstance(sh, list):
            sh = sh[1]  # positive class
        return sh
    return explainer.shap_values(X)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "figures/fig7_shap_summary.png",
    title: str = "SHAP Feature Importance — Best Model",
) -> None:
    """SHAP beeswarm summary plot; red = high feature value, blue = low."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    shap_values = np.atleast_2d(shap_values)
    n_rows, n_feat = shap_values.shape[0], shap_values.shape[1]
    # Use numpy for SHAP to avoid index alignment issues; feature names from X
    X_plot = X.iloc[:n_rows].reset_index(drop=True) if n_rows <= len(X) else X.reset_index(drop=True)
    if X_plot.shape[0] != shap_values.shape[0] or X_plot.shape[1] != shap_values.shape[1]:
        return
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_plot, show=False, max_display=min(20, n_feat))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list,
    save_path: str = "figures/fig8_shap_bar.png",
    title: str = "Average Feature Impact on Long COVID Prediction",
) -> None:
    """Bar plot of mean absolute SHAP value per feature."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    if isinstance(feature_names, (pd.Index, pd.RangeIndex)):
        feature_names = list(feature_names)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), mean_abs[order], color="steelblue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_shap_waterfall(
    shap_values_one: np.ndarray,
    X_one: pd.DataFrame,
    base_value: float,
    save_path: str = "figures/fig8_shap_waterfall.png",
) -> None:
    """Waterfall plot for one sample (first row of SHAP values and X)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_one,
            base_values=base_value,
            data=X_one.values.flatten(),
            feature_names=list(X_one.columns),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def run_shap_for_best_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    figures_dir: str = "figures",
    max_test_samples: int = 1000,
) -> tuple[np.ndarray, str]:
    """
    Build explainer, compute SHAP on test set, save beeswarm and bar plots.
    Returns (shap_values, top_feature_name).
    """
    # Use training set as background for Tree/Linear explainer; sample for Kernel
    if "SVM" in model_name or "Kernel" in str(type(model)):
        X_bg = X_train.sample(min(100, len(X_train)), random_state=42)
    else:
        X_bg = X_train
    explainer = get_explainer(model, X_bg, model_name)
    # Sample test set for SHAP if large (faster); reset index so rows align with SHAP output
    if len(X_test) > max_test_samples:
        X_test_shap = X_test.sample(max_test_samples, random_state=42).reset_index(drop=True)
    else:
        X_test_shap = X_test.reset_index(drop=True).copy()
    shap_values = get_shap_values(explainer, X_test_shap, model_name)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    shap_values = np.atleast_2d(shap_values)
    if shap_values.shape[0] != len(X_test_shap):
        min_len = min(shap_values.shape[0], len(X_test_shap))
        shap_values = shap_values[:min_len]
        X_test_shap = X_test_shap.iloc[:min_len].reset_index(drop=True)
    # Beeswarm
    plot_shap_summary(
        shap_values,
        X_test,
        save_path=os.path.join(figures_dir, "fig7_shap_summary.png"),
        title="SHAP Feature Importance — Best Model",
    )
    # Bar
    plot_shap_bar(
        shap_values,
        list(X_test.columns),
        save_path=os.path.join(figures_dir, "fig8_shap_bar.png"),
        title="Average Feature Impact on Long COVID Prediction",
    )
    # Top feature by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feature = X_test.columns[top_idx]
    # Optional: waterfall for first test sample
    try:
        base = explainer.expected_value
        if isinstance(base, np.ndarray):
            base = base[1]
        plot_shap_waterfall(
            shap_values[0],
            X_test_shap.iloc[[0]],
            base,
            save_path=os.path.join(figures_dir, "fig8_shap_waterfall.png"),
        )
    except Exception:
        pass
    return shap_values, top_feature
