"""
NHSJS Revision — Full ML Rerun
==============================

Reproduces every number the revised manuscript flags as [TODO].

INPUTS
------
Mexico COVID-19 open dataset:
  https://www.kaggle.com/datasets/meirnizri/covid19-dataset
Save as `Covid_Data.csv` in the same directory as this script.

OUTPUTS
-------
results.json                  — every metric the manuscript needs
selected_hyperparameters.json — final tuned hyperparameters per model
subsample_check.csv           — feature-distribution comparison
figures/*.png                 — Fig 3, 4, 5, 6, 6b, 6c, 6d, 6e, 7, 8, 9
saved_models/*.pkl            — trained pipelines

REQUIREMENTS
------------
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib statsmodels scipy

NOTE
----
Random seed fixed at 42 throughout. Total runtime ~30-90 min depending on
hardware (grid search + bootstrap is the bottleneck). Reduce N_BOOTSTRAP or
the grid sizes if you need a faster pass first.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_PATH = "data/raw/Covid Data.csv"
OUT_DIR = Path("revision_outputs")
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "saved_models"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

N_SUBSAMPLE = 25_000
TEST_FRACTION = 0.20
N_FOLDS = 5
N_BOOTSTRAP = 1_000

FEATURES = [
    "AGE", "SEX", "PNEUMONIA", "DIABETES", "ASTHMA",
    "HIPERTENSION", "OBESITY", "CARDIOVASCULAR",
    "RENAL_CHRONIC", "TOBACCO",
]
BINARY_FEATURES = [f for f in FEATURES if f != "AGE"]   # use mode imputation
CONTINUOUS_FEATURES = ["AGE"]                            # use median imputation
SCALE_REQUIRED_MODELS = {"LogisticRegression", "SVM"}    # only these get scaled

# -----------------------------------------------------------------------------
# 1. LOAD AND FILTER
# -----------------------------------------------------------------------------
print("[1/12] Loading data...")
df = pd.read_csv(DATA_PATH)

df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].copy()
print(f"  Confirmed positive cases: {len(df):,}")

# Outcome: hospitalization (PATIENT_TYPE = 2 -> 1, else 0)
df["target"] = (df["PATIENT_TYPE"] == 2).astype(int)

# Replace 97/98/99 with NaN (these are missing-data codes)
for col in FEATURES:
    df[col] = df[col].replace([97, 98, 99], np.nan)

print(f"  Class balance: home={int((df['target']==0).sum()):,} hosp={int((df['target']==1).sum()):,}")

# -----------------------------------------------------------------------------
# 2. STRATIFIED SUBSAMPLE + DISTRIBUTION CHECK
# -----------------------------------------------------------------------------
print("[2/12] Stratified subsample...")
df_sub, _ = train_test_split(
    df, train_size=N_SUBSAMPLE, stratify=df["target"], random_state=RNG,
)

# Compare feature distributions full vs subsample (issue 6)
ks_rows = []
for f in FEATURES:
    full_vals = df[f].dropna().values
    sub_vals = df_sub[f].dropna().values
    if f == "AGE":
        ks_stat, ks_p = stats.ks_2samp(full_vals, sub_vals)
        full_mean, sub_mean = full_vals.mean(), sub_vals.mean()
    else:
        ks_stat, ks_p = stats.ks_2samp(full_vals, sub_vals)
        full_mean, sub_mean = full_vals.mean(), sub_vals.mean()
    ks_rows.append({
        "feature": f, "full_mean": round(full_mean, 4),
        "subsample_mean": round(sub_mean, 4),
        "abs_diff": round(abs(full_mean - sub_mean), 4),
        "ks_statistic": round(ks_stat, 4), "ks_p_value": round(ks_p, 4),
    })
ks_df = pd.DataFrame(ks_rows)
ks_df.to_csv(OUT_DIR / "subsample_check.csv", index=False)
print(f"  Subsample distribution check saved (max KS p = {ks_df['ks_p_value'].max():.4f})")

X = df_sub[FEATURES].copy()
y = df_sub["target"].copy()

# -----------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT (held-out test set untouched until final eval)
# -----------------------------------------------------------------------------
print("[3/12] Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRACTION, stratify=y, random_state=RNG,
)
print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

# -----------------------------------------------------------------------------
# 4. PIPELINE FACTORY (fold-aware imputation + conditional scaling + SMOTE)
# -----------------------------------------------------------------------------
def make_preprocessor(scale: bool):
    """Median imputation for AGE, mode imputation for binary features.
    Scaling only applied if `scale=True` (LR / SVM)."""
    if scale:
        cont_pipe = ImbPipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        bin_pipe = ImbPipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler()),
        ])
    else:
        cont_pipe = SimpleImputer(strategy="median")
        bin_pipe = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(transformers=[
        ("cont", cont_pipe, CONTINUOUS_FEATURES),
        ("bin", bin_pipe, BINARY_FEATURES),
    ])

def make_pipeline(model_name: str, classifier):
    scale = model_name in SCALE_REQUIRED_MODELS
    return ImbPipeline(steps=[
        ("preprocessor", make_preprocessor(scale)),
        ("smote", SMOTE(random_state=RNG)),
        ("clf", classifier),
    ])

# -----------------------------------------------------------------------------
# 5. GRID SEARCH FOR ALL FOUR MODELS
# -----------------------------------------------------------------------------
print("[4/12] Hyperparameter tuning (grid search, 5-fold CV)...")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)

grids = {
    "LogisticRegression": {
        "estimator": LogisticRegression(solver="liblinear", class_weight="balanced",
                                        max_iter=1000, random_state=RNG),
        "param_grid": {
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1, 10],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(class_weight="balanced", bootstrap=True,
                                            random_state=RNG, n_jobs=-1),
        "param_grid": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
    },
    "XGBoost": {
        "estimator": xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            random_state=RNG, n_jobs=-1, use_label_encoder=False,
        ),
        "param_grid": {
            "clf__n_estimators": [100, 300, 500],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__gamma": [0, 0.1],
            "clf__reg_lambda": [1, 10],
            "clf__reg_alpha": [0, 1],
        },
    },
    "SVM": {
        "estimator": SVC(kernel="rbf", probability=True, random_state=RNG),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
    },
}

# To keep XGBoost grid tractable, take a randomized subset (still much wider
# than the original "default-only" approach). Toggle to full grid if you have
# the compute budget.
USE_RANDOMIZED_FOR_XGB = True

selected_hyperparams = {}
fitted_pipelines = {}
cv_aucs = {}

for name, cfg in grids.items():
    print(f"  Tuning {name}...")
    pipe = make_pipeline(name, cfg["estimator"])

    if name == "XGBoost" and USE_RANDOMIZED_FOR_XGB:
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(
            pipe, cfg["param_grid"], n_iter=40, cv=cv,
            scoring="roc_auc", n_jobs=-1, random_state=RNG, refit=True,
        )
    else:
        search = GridSearchCV(
            pipe, cfg["param_grid"], cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True,
        )

    search.fit(X_train, y_train)
    selected_hyperparams[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    fitted_pipelines[name] = search.best_estimator_
    cv_aucs[name] = round(float(search.best_score_), 4)
    print(f"    best CV AUC = {cv_aucs[name]:.4f}")

with open(OUT_DIR / "selected_hyperparameters.json", "w") as f:
    json.dump({"selected_params": selected_hyperparams, "cv_aucs": cv_aucs}, f, indent=2)

# -----------------------------------------------------------------------------
# 6. EVAL HELPERS
# -----------------------------------------------------------------------------
def eval_at_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "brier": round(brier_score_loss(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def cv_fold_aucs(pipe, X, y):
    """Return per-fold AUC for stability reporting."""
    aucs = []
    for tr, va in cv.split(X, y):
        pipe_clone = type(pipe)(steps=[(n, type(s)(**s.get_params())) if hasattr(s, "get_params") else (n, s) for n, s in pipe.steps])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], proba))
    return aucs

def bootstrap_ci(y_true, y_proba, threshold, metric_fn, n=N_BOOTSTRAP, ci=95):
    rng = np.random.default_rng(RNG)
    n_obs = len(y_true)
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            vals.append(metric_fn(y_true_arr[idx], y_proba_arr[idx], threshold))
        except ValueError:
            continue
    lo = np.percentile(vals, (100 - ci) / 2)
    hi = np.percentile(vals, 100 - (100 - ci) / 2)
    return round(float(lo), 4), round(float(hi), 4)

def metric_auc(y_t, y_p, thr): return roc_auc_score(y_t, y_p)
def metric_precision(y_t, y_p, thr): return precision_score(y_t, (y_p >= thr).astype(int), zero_division=0)
def metric_recall(y_t, y_p, thr): return recall_score(y_t, (y_p >= thr).astype(int), zero_division=0)

def delong_test(y_true, proba_a, proba_b):
    """DeLong's test for two correlated AUCs.
    Returns (z_statistic, p_value)."""
    from scipy.stats import norm
    y = np.asarray(y_true)
    pos = y == 1
    neg = y == 0
    n1, n0 = pos.sum(), neg.sum()

    def midrank(x):
        order = np.argsort(x)
        ranked = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and x[order[j]] == x[order[j + 1]]:
                j += 1
            avg = 0.5 * (i + j) + 1
            for k in range(i, j + 1):
                ranked[order[k]] = avg
            i = j + 1
        return ranked

    aucs = []
    V10s = []
    V01s = []
    for proba in (proba_a, proba_b):
        proba = np.asarray(proba)
        tx = proba[pos]
        ty = proba[neg]
        tz = np.concatenate([tx, ty])
        tx_rank = midrank(tx)
        ty_rank = midrank(ty)
        tz_rank = midrank(tz)
        auc = (tz_rank[:n1].sum() / (n1 * n0)) - (n1 + 1) / (2 * n0)
        aucs.append(auc)
        V10 = (tz_rank[:n1] - tx_rank) / n0
        V01 = 1 - (tz_rank[n1:] - ty_rank) / n1
        V10s.append(V10)
        V01s.append(V01)
    aucs = np.array(aucs)
    V10s = np.vstack(V10s)
    V01s = np.vstack(V01s)
    S10 = np.cov(V10s)
    S01 = np.cov(V01s)
    S = S10 / n1 + S01 / n0
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)

# -----------------------------------------------------------------------------
# 7. MAIN MODEL EVALUATION ON HELD-OUT TEST
# -----------------------------------------------------------------------------
print("[5/12] Test-set evaluation for all four models...")
test_probas = {}
test_metrics = {}

for name, pipe in fitted_pipelines.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    test_probas[name] = proba
    m = eval_at_threshold(y_test, proba, threshold=0.5)
    auc_lo, auc_hi = bootstrap_ci(y_test, proba, 0.5, metric_auc)
    pr_lo, pr_hi = bootstrap_ci(y_test, proba, 0.5, metric_precision)
    rc_lo, rc_hi = bootstrap_ci(y_test, proba, 0.5, metric_recall)
    m.update({
        "cv_auc": cv_aucs[name],
        "auc_95ci": [auc_lo, auc_hi],
        "precision_95ci": [pr_lo, pr_hi],
        "recall_95ci": [rc_lo, rc_hi],
    })
    test_metrics[name] = m
    print(f"  {name:18s} AUC={m['roc_auc']:.4f} (95% CI {auc_lo:.4f}-{auc_hi:.4f})  Brier={m['brier']:.4f}")

# -----------------------------------------------------------------------------
# 8. PAIRWISE DELONG TESTS
# -----------------------------------------------------------------------------
print("[6/12] Pairwise DeLong tests...")
delong = {}
names = list(test_probas.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        z, p = delong_test(y_test, test_probas[names[i]], test_probas[names[j]])
        delong[f"{names[i]} vs {names[j]}"] = {"z": round(z, 4), "p_value": round(p, 6)}
        print(f"  {names[i]:14s} vs {names[j]:14s}  z={z:+.3f} p={p:.4f}")

# -----------------------------------------------------------------------------
# 9. BASELINES (rule-based + minimal LR)
# -----------------------------------------------------------------------------
print("[7/12] Baselines...")

# (a) Rule-based: age > 60 OR pneumonia recorded
# Fill missing with 0 (i.e. assume no pneumonia / age threshold not met)
def rule_baseline_predict(X_df):
    age = X_df["AGE"].fillna(X_train["AGE"].median()).values
    pneu = X_df["PNEUMONIA"].fillna(0).values
    pneu_pos = (pneu == 1).astype(int)
    return ((age > 60) | (pneu_pos == 1)).astype(int)

# Pseudo-probability: 1.0 if rule fires, 0.0 otherwise (degenerate but evaluable)
y_pred_rule = rule_baseline_predict(X_test)
y_proba_rule = y_pred_rule.astype(float)
rule_metrics = eval_at_threshold(y_test, y_proba_rule, threshold=0.5)

# (b) Minimal LR: age + pneumonia only
X_train_min = X_train[["AGE", "PNEUMONIA"]].copy()
X_test_min = X_test[["AGE", "PNEUMONIA"]].copy()

mini_pipe = ImbPipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=[
        ("cont", ImbPipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler())]), ["AGE"]),
        ("bin", ImbPipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                   ("scale", StandardScaler())]), ["PNEUMONIA"]),
    ])),
    ("smote", SMOTE(random_state=RNG)),
    ("clf", LogisticRegression(solver="liblinear", class_weight="balanced",
                                max_iter=1000, random_state=RNG)),
])
mini_pipe.fit(X_train_min, y_train)
y_proba_mini = mini_pipe.predict_proba(X_test_min)[:, 1]
mini_metrics = eval_at_threshold(y_test, y_proba_mini, threshold=0.5)
mini_auc_lo, mini_auc_hi = bootstrap_ci(y_test, y_proba_mini, 0.5, metric_auc)
mini_metrics["auc_95ci"] = [mini_auc_lo, mini_auc_hi]

baselines = {"rule_based": rule_metrics, "minimal_lr_age_pneumonia": mini_metrics}
print(f"  Rule-based AUC = {rule_metrics['roc_auc']:.4f}")
print(f"  Minimal LR AUC = {mini_metrics['roc_auc']:.4f} (95% CI {mini_auc_lo:.4f}-{mini_auc_hi:.4f})")

# DeLong: minimal LR vs each main model
delong_vs_minimal = {}
for name, proba in test_probas.items():
    z, p = delong_test(y_test, proba, y_proba_mini)
    delong_vs_minimal[f"{name} vs minimal_LR"] = {"z": round(z, 4), "p_value": round(p, 6)}

# -----------------------------------------------------------------------------
# 10. SENSITIVITY ANALYSIS WITHOUT PNEUMONIA
# -----------------------------------------------------------------------------
print("[8/12] Sensitivity analysis (no pneumonia)...")
FEATURES_NO_PNEU = [f for f in FEATURES if f != "PNEUMONIA"]
X_train_np = X_train[FEATURES_NO_PNEU]
X_test_np = X_test[FEATURES_NO_PNEU]

def make_preprocessor_no_pneu(scale: bool):
    bin_no_pneu = [f for f in BINARY_FEATURES if f != "PNEUMONIA"]
    if scale:
        cont_pipe = ImbPipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())])
        bin_pipe = ImbPipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                      ("scale", StandardScaler())])
    else:
        cont_pipe = SimpleImputer(strategy="median")
        bin_pipe = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(transformers=[
        ("cont", cont_pipe, CONTINUOUS_FEATURES),
        ("bin", bin_pipe, bin_no_pneu),
    ])

sensitivity_results = {}
for name, cfg in grids.items():
    pipe = ImbPipeline(steps=[
        ("preprocessor", make_preprocessor_no_pneu(name in SCALE_REQUIRED_MODELS)),
        ("smote", SMOTE(random_state=RNG)),
        ("clf", cfg["estimator"]),
    ])
    # Use the same selected hyperparameters from the full-feature run
    params = {f"clf__{k}": v for k, v in selected_hyperparams[name].items()}
    pipe.set_params(**params)
    pipe.fit(X_train_np, y_train)
    proba = pipe.predict_proba(X_test_np)[:, 1]
    m = eval_at_threshold(y_test, proba, threshold=0.5)
    sensitivity_results[name] = m
    print(f"  {name:18s} (no pneumonia) AUC={m['roc_auc']:.4f}")

# -----------------------------------------------------------------------------
# 11. THRESHOLD OPTIMIZATION + PR + F1 + CALIBRATION + DCA (XGBoost = top model)
# -----------------------------------------------------------------------------
print("[9/12] Threshold + PR + F1 + calibration + DCA for XGBoost...")
xgb_proba = test_probas["XGBoost"]

# F1-maximizing threshold
thresholds = np.linspace(0.05, 0.95, 91)
f1s = [f1_score(y_test, (xgb_proba >= t).astype(int), zero_division=0) for t in thresholds]
best_t_idx = int(np.argmax(f1s))
best_threshold = float(thresholds[best_t_idx])
print(f"  F1-max threshold = {best_threshold:.3f}  F1 = {f1s[best_t_idx]:.4f}")

xgb_at_best_t = eval_at_threshold(y_test, xgb_proba, threshold=best_threshold)

# Precision-recall
precisions, recalls, _ = precision_recall_curve(y_test, xgb_proba)
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, color="C0", lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("XGBoost — Precision-Recall Curve (test set)")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6b_pr_curve.png", dpi=200); plt.close()

# F1 vs threshold
plt.figure(figsize=(6, 5))
plt.plot(thresholds, f1s, color="C2", lw=2)
plt.axvline(best_threshold, color="red", ls="--", label=f"F1-max @ {best_threshold:.2f}")
plt.axvline(0.5, color="gray", ls=":", label="Default 0.5")
plt.xlabel("Decision threshold"); plt.ylabel("F1-score")
plt.title("XGBoost — F1 vs Threshold (test set)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6c_f1_vs_threshold.png", dpi=200); plt.close()

# Calibration
prob_true, prob_pred = calibration_curve(y_test, xgb_proba, n_bins=10, strategy="quantile")
plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.plot(prob_pred, prob_true, marker="o", color="C0", label="XGBoost")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction positive")
plt.title("XGBoost — Calibration Curve (test set)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6d_calibration.png", dpi=200); plt.close()

# Decision-curve analysis
def net_benefit(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if threshold >= 1:  # avoid division by zero
        return 0.0
    return tp / n - fp / n * (threshold / (1 - threshold))

dca_thresholds = np.linspace(0.05, 0.50, 46)
nb_model = [net_benefit(y_test, xgb_proba, t) for t in dca_thresholds]
nb_all = []
prevalence = float(y_test.mean())
for t in dca_thresholds:
    if t >= 1:
        nb_all.append(0.0)
    else:
        nb_all.append(prevalence - (1 - prevalence) * (t / (1 - t)))
nb_none = [0.0] * len(dca_thresholds)

plt.figure(figsize=(6, 5))
plt.plot(dca_thresholds, nb_model, lw=2, label="XGBoost", color="C0")
plt.plot(dca_thresholds, nb_all, lw=1, ls="--", label="Treat all", color="C1")
plt.plot(dca_thresholds, nb_none, lw=1, ls=":", label="Treat none", color="gray")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("XGBoost — Decision-Curve Analysis (test set)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6e_dca.png", dpi=200); plt.close()

# Threshold range with positive net benefit (vs treat-all and treat-none)
positive_thresholds = [
    float(t) for t, nbm, nba in zip(dca_thresholds, nb_model, nb_all)
    if (nbm > nba) and (nbm > 0)
]
dca_range = (
    [round(min(positive_thresholds), 3), round(max(positive_thresholds), 3)]
    if positive_thresholds else None
)
print(f"  DCA positive-net-benefit range: {dca_range}")

# Calibration interpretation
calibration_status = "well-calibrated"
mean_diff = float(np.mean(prob_pred - prob_true))
if mean_diff > 0.05:
    calibration_status = "mild over-confidence"
elif mean_diff < -0.05:
    calibration_status = "mild under-confidence"

# -----------------------------------------------------------------------------
# 12. FIGURE 3, 4, 5, 6, 7, 8, 9
# -----------------------------------------------------------------------------
print("[10/12] Generating figures 3-9...")

# Fig 3 — bar chart of acc/F1/AUC across all models + baselines
fig3_models = ["Rule", "MinLR", "LR", "RF", "XGB", "SVM"]
fig3_data = [
    [rule_metrics["accuracy"], rule_metrics["f1"], rule_metrics["roc_auc"]],
    [mini_metrics["accuracy"], mini_metrics["f1"], mini_metrics["roc_auc"]],
    [test_metrics["LogisticRegression"]["accuracy"], test_metrics["LogisticRegression"]["f1"], test_metrics["LogisticRegression"]["roc_auc"]],
    [test_metrics["RandomForest"]["accuracy"], test_metrics["RandomForest"]["f1"], test_metrics["RandomForest"]["roc_auc"]],
    [test_metrics["XGBoost"]["accuracy"], test_metrics["XGBoost"]["f1"], test_metrics["XGBoost"]["roc_auc"]],
    [test_metrics["SVM"]["accuracy"], test_metrics["SVM"]["f1"], test_metrics["SVM"]["roc_auc"]],
]
arr = np.array(fig3_data)
x = np.arange(len(fig3_models)); w = 0.27
plt.figure(figsize=(9, 5))
plt.bar(x - w, arr[:, 0], w, label="Accuracy")
plt.bar(x, arr[:, 1], w, label="F1")
plt.bar(x + w, arr[:, 2], w, label="ROC-AUC")
plt.xticks(x, fig3_models); plt.ylim(0, 1.0); plt.legend()
plt.title("Figure 3 — Model + Baseline Comparison")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_model_comparison.png", dpi=200); plt.close()

# Fig 4 — ROC
plt.figure(figsize=(6, 5))
for name, proba in test_probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {test_metrics[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Figure 4 — ROC Curves"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_roc.png", dpi=200); plt.close()

# Fig 5 — CV fold AUC distribution
print("  Computing per-fold AUCs for Figure 5...")
fold_aucs_per_model = {}
for name, pipe in fitted_pipelines.items():
    fold_aucs = []
    for tr, va in cv.split(X_train, y_train):
        # Refit a fresh clone for each fold
        from sklearn.base import clone
        cl = clone(pipe)
        cl.fit(X_train.iloc[tr], y_train.iloc[tr])
        proba = cl.predict_proba(X_train.iloc[va])[:, 1]
        fold_aucs.append(roc_auc_score(y_train.iloc[va], proba))
    fold_aucs_per_model[name] = fold_aucs
    print(f"    {name:18s} fold AUCs: {[round(a, 4) for a in fold_aucs]}  std={np.std(fold_aucs):.4f}")

plt.figure(figsize=(7, 5))
plt.boxplot([fold_aucs_per_model[n] for n in test_probas.keys()],
            labels=list(test_probas.keys()))
plt.ylabel("CV ROC-AUC"); plt.title("Figure 5 — Cross-Validation AUC Distribution")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_cv_distribution.png", dpi=200); plt.close()

# Fig 6 — Confusion matrix for XGBoost @ default 0.5
cm = np.array(test_metrics["XGBoost"]["confusion_matrix"])
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=14)
plt.xticks([0, 1], ["Home", "Hosp"]); plt.yticks([0, 1], ["Home", "Hosp"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Figure 6 — XGBoost Confusion Matrix (threshold = 0.5)")
plt.colorbar()
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_confusion_matrix.png", dpi=200); plt.close()

# SHAP figures (Fig 7, 8, 9) for XGBoost
print("  Generating SHAP figures...")
xgb_pipe = fitted_pipelines["XGBoost"]
# Pull the fitted classifier and a transformed sample of test data
preproc = xgb_pipe.named_steps["preprocessor"]
X_test_transformed = preproc.transform(X_test)
clf = xgb_pipe.named_steps["clf"]
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_transformed)

# Fig 7 — beeswarm
plt.figure()
shap.summary_plot(shap_values, X_test_transformed, feature_names=FEATURES,
                  show=False, plot_type="dot")
plt.title("Figure 7 — SHAP Beeswarm (XGBoost)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig7_shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 8 — mean absolute SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]
plt.figure(figsize=(7, 5))
plt.barh([FEATURES[i] for i in order][::-1], [mean_abs_shap[i] for i in order][::-1])
plt.xlabel("Mean |SHAP value|"); plt.title("Figure 8 — Mean Absolute SHAP")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig8_shap_meanabs.png", dpi=200); plt.close()

shap_means = {FEATURES[i]: round(float(mean_abs_shap[i]), 4) for i in order}
print(f"  SHAP mean |value|: {shap_means}")

# Fig 9 — waterfall for one high-risk patient
high_risk_idx = int(np.argmax(xgb_proba))
plt.figure()
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_values[high_risk_idx],
    feature_names=FEATURES, show=False, max_display=10,
)
plt.title(f"Figure 9 — SHAP Waterfall (patient idx {high_risk_idx})")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig9_shap_waterfall.png", dpi=200, bbox_inches="tight"); plt.close()

# -----------------------------------------------------------------------------
# 13. CONSOLIDATE EVERYTHING -> results.json
# -----------------------------------------------------------------------------
print("[11/12] Writing results.json...")
results = {
    "subsample_check": {
        "n_full_filtered": int(len(df)),
        "n_subsample": int(N_SUBSAMPLE),
        "max_ks_p_value": float(ks_df["ks_p_value"].max()),
        "max_abs_mean_diff": float(ks_df["abs_diff"].max()),
        "feature_table_csv": "subsample_check.csv",
    },
    "selected_hyperparameters": selected_hyperparams,
    "cv_auc_per_model": cv_aucs,
    "cv_fold_aucs": {k: [round(v, 4) for v in vals] for k, vals in fold_aucs_per_model.items()},
    "cv_fold_std": {k: round(float(np.std(v)), 4) for k, v in fold_aucs_per_model.items()},
    "test_metrics_default_threshold": test_metrics,
    "baselines": baselines,
    "delong_pairwise_main": delong,
    "delong_main_vs_minimal_lr": delong_vs_minimal,
    "sensitivity_no_pneumonia": sensitivity_results,
    "xgboost_threshold_analysis": {
        "default_threshold": test_metrics["XGBoost"],
        "f1_maximizing_threshold": best_threshold,
        "metrics_at_f1_max": xgb_at_best_t,
    },
    "calibration": {
        "brier_score_xgboost": test_metrics["XGBoost"]["brier"],
        "interpretation": calibration_status,
        "mean_pred_minus_obs": round(mean_diff, 4),
    },
    "decision_curve_analysis": {
        "positive_net_benefit_threshold_range": dca_range,
        "thresholds_evaluated": [float(round(t, 3)) for t in dca_thresholds.tolist()],
    },
    "shap_mean_abs": shap_means,
}

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save fitted pipelines
print("[12/12] Saving fitted pipelines...")
import pickle
for name, pipe in fitted_pipelines.items():
    with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(pipe, f)

print("\nDone.")
print(f"  All outputs in: {OUT_DIR.resolve()}")
print(f"  Insert numbers from results.json into the [TODO] placeholders in")
print(f"  Revised_Manuscript_Clean.docx and Revised_Manuscript_TrackedChanges.docx.")
