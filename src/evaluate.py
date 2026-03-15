"""
Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# Use a consistent style and 300 dpi for publication figures
FIGURE_DPI = 300
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = FIGURE_DPI


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Compute Accuracy, Precision, Recall, F1, ROC-AUC (if y_proba provided)."""
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    else:
        out["ROC_AUC"] = np.nan
    return out


def evaluate_all_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: dict,
) -> pd.DataFrame:
    """
    For each model: predict and get proba (if available), compute metrics.
    Returns a DataFrame with columns: Model, CV_Accuracy, Test_Accuracy, Precision, Recall, F1_Score, ROC_AUC.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
        m = compute_metrics(y_test.values, y_pred, y_proba)
        rows.append({
            "Model": name,
            "CV_Accuracy": cv_results.get(name, np.nan),
            "Test_Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1_Score": m["F1_Score"],
            "ROC_AUC": m["ROC_AUC"],
        })
    return pd.DataFrame(rows)


def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    save_path: str = "figures/fig4_roc_curves.png",
) -> None:
    """Plot ROC curves for all models on one graph; dashed diagonal; AUC in legend."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random chance")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, model), color in zip(models.items(), colors):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {auc:.2f})")
        except Exception:
            continue
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "figures/fig5_confusion_matrix.png",
    title: str = "Confusion Matrix — Best Model",
) -> None:
    """Seaborn heatmap confusion matrix with True/Predicted labels."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_cv_accuracy_box(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    save_path: str = "figures/fig6_cv_accuracy.png",
) -> None:
    """Box plot of 5-fold CV accuracy for each model."""
    from sklearn.model_selection import cross_val_score
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = []
    scores_list = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        names.append(name)
        scores_list.append(scores)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(scores_list, labels=names, patch_artist=True)
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Validation Accuracy Distribution by Model")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save_path: str = "figures/fig3_model_comparison.png",
) -> None:
    """Grouped bar chart: Test Accuracy, F1 Score, ROC-AUC for all models; value labels on bars."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    x = np.arange(len(metrics_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics_df["Test_Accuracy"], width, label="Test Accuracy", color="#1f77b4")
    ax.bar(x, metrics_df["F1_Score"], width, label="F1 Score", color="#ff7f0e")
    ax.bar(x + width, metrics_df["ROC_AUC"], width, label="ROC-AUC", color="#2ca02c")
    ax.set_ylabel("Score")
    ax.set_title("ML Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, row in metrics_df.iterrows():
        ax.text(i - width, row["Test_Accuracy"] + 0.02, f"{row['Test_Accuracy']:.2f}", ha="center", fontsize=8)
        ax.text(i, row["F1_Score"] + 0.02, f"{row['F1_Score']:.2f}", ha="center", fontsize=8)
        ax.text(i + width, row["ROC_AUC"] + 0.02, f"{row['ROC_AUC']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
