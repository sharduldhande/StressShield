"""
retrain_no_eda.py — retrain the stress model using only ECG + respiration features.

Reads wesad_features.csv, drops EDA columns, retrains XGBoost with LOSO
cross-validation, then saves stress_model_no_eda.pkl.

Usage:
    python retrain_no_eda.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

FEATURES_CSV = "wesad_features.csv"
OUTPUT_MODEL  = "stress_model_no_eda.pkl"

EDA_COLS = [
    "eda_tonic_mean", "eda_tonic_std", "eda_tonic_slope",
    "eda_phasic_mean", "eda_phasic_std", "eda_scr_count",
]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df.drop(columns=EDA_COLS, errors="ignore")
    print(f"Loaded {len(df)} windows, {df['label'].mean():.1%} stress")
    return df


def run_loso(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].values
    y = df["label"].values
    groups = df["subject"].values

    all_true, all_pred, all_prob = [], [], []

    print("\n── LOSO cross-validation ────────────────────────────")
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        subj = np.unique(groups[test_idx])[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        mask_tr = ~np.isnan(X_tr).any(1)
        mask_te = ~np.isnan(X_te).any(1)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
        X_te, y_te = X_te[mask_te], y_te[mask_te]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42,
        )
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
        auc   = roc_auc_score(y_te, probs)
        print(f"  [{subj}]  AUC: {auc:.3f}")

        all_true.extend(y_te)
        all_pred.extend(preds)
        all_prob.extend(probs)

    print("\n── Overall ───────────────────────────────────────────")
    print(classification_report(all_true, all_pred,
                                target_names=["Non-stress", "Stress"]))
    print(f"ROC-AUC: {roc_auc_score(all_true, all_prob):.3f}")


def train_final(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].values
    y = df["label"].values
    mask = ~np.isnan(X).any(1)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=42,
    )
    clf.fit(X_s, y)

    joblib.dump({"model": clf, "scaler": scaler, "features": feat_cols}, OUTPUT_MODEL)
    print(f"\nModel saved → {OUTPUT_MODEL}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")


def main():
    df = load_data()
    feat_cols = [c for c in df.columns if c not in ("label", "subject")]
    run_loso(df, feat_cols)
    train_final(df, feat_cols)


if __name__ == "__main__":
    main()
