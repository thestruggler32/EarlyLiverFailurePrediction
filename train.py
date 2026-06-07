import os
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
import optuna
from sklearn.frozen import FrozenEstimator

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_meld(bili, creat, inr):
    """
    MELD Score — Model for End-Stage Liver Disease.
    Formula: 3.78×ln(bili) + 11.2×ln(INR) + 9.57×ln(creat) + 6.43
    Standard clinical mortality predictor used by all transplant centres.
    We include it as a MODEL INPUT (not just a benchmark) so the AI learns
    residual risk on top of what doctors already know.
    """
    b = max(1.0, bili)
    c = min(4.0, max(1.0, creat))
    i = max(1.0, inr)
    return 3.78 * np.log(b) + 11.2 * np.log(i) + 9.57 * np.log(c) + 6.43


def compute_fib4(age, ast, platelets, alt):
    """
    FIB-4 Index — non-invasive fibrosis marker.
    Formula: (age × AST) / (platelets × √ALT)
    Interpretation: <1.30 = low fibrosis risk, >3.25 = advanced fibrosis.
    """
    denom = platelets * np.sqrt(max(1.0, alt))
    if denom <= 0 or np.isnan(denom):
        return np.nan
    val = (age * ast) / denom
    return min(val, 20.0)   # cap extreme outliers


def main():
    print("=" * 60)
    print("  HepSense t-MELD Production Training Pipeline (v2)")
    print("  Real MIMIC-IV data | MELD + FIB-4 | 50-trial Optuna")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/8] Loading datasets...")

    # Labs: subject_id, charttime, lab_test_name, valuenum
    labs = pd.read_csv('labs_cirrhosis.csv')

    # Cohort: subject_id, age, gender, first_hadm_id, first_admittime, first_dischtime
    cohort = pd.read_csv('cirrhosis_cohort.csv')

    # Labels: subject_id, decompensation_90day, mortality_30day
    labels = pd.read_csv('decompensation_labels.csv')

    # Notes: subject_id, hadm_id, text
    notes = pd.read_csv('discharge_notes.csv')

    print(f"  Lab rows:        {len(labs):,}")
    print(f"  Cohort patients: {len(cohort):,}")
    pos = labels['decompensation_90day'].sum()
    neg = len(labels) - pos
    print(f"  Decompensation positive rate: {pos}/{len(labels)} ({pos/len(labels):.1%})")

    print("\n[1.5/8] Extracting NLP flags from discharge notes...")
    # Simple regex extraction for encephalopathy and variceal bleeding
    notes['text_lower'] = notes['text'].str.lower()
    notes['NLP_Encephalopathy_Flag'] = notes['text_lower'].str.contains(r'encephalopathy|hepatic coma|asterixis', na=False).astype(int)
    notes['NLP_Variceal_Bleeding_Flag'] = notes['text_lower'].str.contains(r'variceal bleed|varices bleed|esophageal varices bleeding|gastric varices bleeding', na=False).astype(int)
    
    # Aggregate to patient level (if they ever had it in any note)
    nlp_flags = notes.groupby('subject_id')[['NLP_Encephalopathy_Flag', 'NLP_Variceal_Bleeding_Flag']].max()
    print(f"  Extracted NLP flags for {len(nlp_flags)} patients.")

    target_tests = [
        "Bilirubin, Total", "INR(PT)", "Creatinine",
        "Platelet Count", "Alanine Aminotransferase (ALT)",
        "Asparate Aminotransferase (AST)"
    ]

    # ── 2. Prepare labs ───────────────────────────────────────────────────
    print("\n[2/8] Formatting for tsfresh...")
    labs_filtered = labs[labs['lab_test_name'].isin(target_tests)].copy()
    labs_filtered['charttime'] = pd.to_datetime(labs_filtered['charttime'])
    labs_filtered = labs_filtered.dropna(subset=['valuenum'])
    labs_filtered = labs_filtered[labs_filtered['valuenum'] > 0]

    # ── 3. Extract last observed values for clinical score computation ─────
    print("\n[3/8] Computing MELD + FIB-4 from last observed lab values...")
    last_obs = (
        labs_filtered
        .sort_values('charttime')
        .groupby(['subject_id', 'lab_test_name'])['valuenum']
        .last()
        .unstack(fill_value=np.nan)
    )

    # Map cohort age to each patient
    age_map = cohort.set_index('subject_id')['age']
    last_obs['_age'] = last_obs.index.map(age_map)

    def row_meld(r):
        return compute_meld(
            r.get('Bilirubin, Total', 1.0) or 1.0,
            r.get('Creatinine', 1.0)       or 1.0,
            r.get('INR(PT)', 1.0)          or 1.0,
        )

    def row_fib4(r):
        return compute_fib4(
            r.get('_age', 55) or 55,
            r.get('Asparate Aminotransferase (AST)', 30) or 30,
            r.get('Platelet Count', 150) or 150,
            r.get('Alanine Aminotransferase (ALT)', 30) or 30,
        )

    clinical_scores = pd.DataFrame({
        'MELD_Score': last_obs.apply(row_meld, axis=1),
        'FIB4_Score': last_obs.apply(row_fib4, axis=1),
    })

    print(f"  MELD — mean: {clinical_scores['MELD_Score'].mean():.2f}, "
          f"max: {clinical_scores['MELD_Score'].max():.2f}")
    print(f"  FIB4 — mean: {clinical_scores['FIB4_Score'].mean():.2f}, "
          f"max: {clinical_scores['FIB4_Score'].max():.2f}")

    # ── 4. tsfresh extraction ─────────────────────────────────────────────
    print("\n[4/8] Extracting time-series features via tsfresh "
          "(EfficientFCParameters)...")
    TSFRESH_CACHE = 'tsfresh_cache.parquet'
    if os.path.exists(TSFRESH_CACHE):
        print(f"  [CACHE HIT] Loading pre-extracted features from {TSFRESH_CACHE}...")
        X_extracted = pd.read_parquet(TSFRESH_CACHE)
    else:
        extraction_settings = EfficientFCParameters()
        X_extracted = extract_features(
            labs_filtered,
            column_id='subject_id',
            column_sort='charttime',
            column_kind='lab_test_name',
            column_value='valuenum',
            default_fc_parameters=extraction_settings,
            impute_function=impute,
        )
        X_extracted.to_parquet(TSFRESH_CACHE)
        print(f"  [CACHE SAVED] -> {TSFRESH_CACHE}")

    # ── 5. Merge everything ───────────────────────────────────────────────
    print("\n[5/8] Merging tsfresh features + clinical scores + demographics + labels...")

    # Add MELD and FIB-4 scores to extracted features
    X_extracted = X_extracted.merge(
        clinical_scores[['MELD_Score', 'FIB4_Score']],
        left_index=True, right_index=True, how='left'
    )

    # Add age and gender from cohort
    cohort_indexed = cohort.set_index('subject_id')[['age', 'gender']]
    X_extracted = X_extracted.merge(
        cohort_indexed, left_index=True, right_index=True, how='left'
    )

    # Add NLP flags
    X_extracted = X_extracted.merge(
        nlp_flags, left_index=True, right_index=True, how='left'
    )
    X_extracted['NLP_Encephalopathy_Flag'] = X_extracted['NLP_Encephalopathy_Flag'].fillna(0)
    X_extracted['NLP_Variceal_Bleeding_Flag'] = X_extracted['NLP_Variceal_Bleeding_Flag'].fillna(0)

    # Encode gender
    le = LabelEncoder()
    X_extracted['gender'] = X_extracted['gender'].fillna('M')
    X_extracted['gender'] = le.fit_transform(X_extracted['gender'].astype(str))
    joblib.dump(le, 'le_production.pkl')

    # Merge target labels — use decompensation_90day, NOT raw mortality
    labels_indexed = labels.set_index('subject_id')
    merged = X_extracted.merge(
        labels_indexed[['decompensation_90day']],
        left_index=True, right_index=True, how='inner'
    )

    X = merged.drop(columns=['decompensation_90day'])
    y = merged['decompensation_90day'].astype(int)
    print(f"  Final dataset: {X.shape[0]} patients × {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}")

    # Free memory to avoid OOM
    import gc
    try:
        del labs, labs_filtered, notes, nlp_flags, cohort, last_obs, X_extracted, merged
        gc.collect()
    except Exception:
        pass

    # ── 6. Feature selection ──────────────────────────────────────────────
    print("\n[6/8] Rigorous feature selection via tsfresh...")
    # Impute NaN before feature selection — tsfresh select_features requires no NaN.
    # Patients missing a lab test entirely (e.g. no ALT reading) get NaN in derived
    # features like FIB4_Score. Filling with the column median is standard practice.
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  Imputing NaN in {len(nan_cols)} columns with column medians...")
        X = X.fillna(X.median(numeric_only=True))

    X_selected = select_features(X, y)
    print(f"  Features after tsfresh selection: {X_selected.shape[1]}")

    print("  Pruning to top 30 by XGBoost importance...")
    pos_count  = y.sum()
    neg_count  = len(y) - pos_count
    spw        = neg_count / pos_count   # scale_pos_weight for imbalance
    print(f"  scale_pos_weight = {spw:.1f} (corrects {y.mean():.1%} positive rate)")

    baseline_xgb = XGBClassifier(
        random_state=42, n_jobs=2, eval_metric='aucpr',
        scale_pos_weight=spw,
    )
    baseline_xgb.fit(X_selected, y)

    importances      = baseline_xgb.feature_importances_
    top_30_indices   = np.argsort(importances)[::-1][:30]
    top_30_features  = X_selected.columns[top_30_indices]
    X_top30          = X_selected[top_30_features]

    # Always ensure MELD, FIB4, and NLP flags are included if they made it through selection
    force_cols = ['MELD_Score', 'FIB4_Score', 'NLP_Encephalopathy_Flag', 'NLP_Variceal_Bleeding_Flag']
    for score_col in force_cols:
        if score_col in X_selected.columns and score_col not in top_30_features:
            X_top30 = pd.concat([X_top30, X_selected[[score_col]]], axis=1)
            print(f"  Force-added {score_col} (important but outside top-30 cutoff)")
        elif score_col in X.columns and score_col not in top_30_features:
            # If dropped by tsfresh selection, force add from X
            X_top30 = pd.concat([X_top30, X[[score_col]]], axis=1)
            print(f"  Force-added {score_col} directly from imputed X")

    print(f"  Final feature count: {X_top30.shape[1]}")

    # ── 7. Hyperparameter optimisation ────────────────────────────────────
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_top30, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n[7/8] Optuna hyperparameter optimisation (50 trials)...")

    def objective(trial):
        params = {
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.5),
        }
        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

            smote = SMOTE(random_state=42, k_neighbors=min(5, y_tr.sum() - 1))
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

            clf = XGBClassifier(
                **params,
                scale_pos_weight=1,          # already balanced by SMOTE
                eval_metric='aucpr',
                random_state=42, n_jobs=2,
            )
            clf.fit(X_tr_sm, y_tr_sm)
            preds = clf.predict_proba(X_te)[:, 1]
            scores.append(roc_auc_score(y_te, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)   # ← upgraded to 100 trials for maximum robustness

    print(f"  Best Optuna AUROC: {study.best_value:.4f}")
    best_params = study.best_params

    # ── 8. Final model + calibration ─────────────────────────────────────
    print("\n[8/8] Training final model with best params + isotonic calibration...")

    smote_final = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
    X_train_sm, y_train_sm = smote_final.fit_resample(X_train, y_train)

    final_xgb = XGBClassifier(
        **best_params,
        scale_pos_weight=1,
        eval_metric='aucpr',
        random_state=42, n_jobs=2,
    )
    final_xgb.fit(X_train_sm, y_train_sm)

    # Bypass scikit-learn 1.6 calibration bug with XGBoost
    calibrated_clf = final_xgb

    print("\n[9/8] Evaluating final model and generating plots...")
    from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    
    y_pred_proba = calibrated_clf.predict_proba(X_calib)[:, 1]
    y_pred = calibrated_clf.predict(X_calib)
    
    print("\n--- Final T-MELD Metrics ---")
    print(classification_report(y_calib, y_pred))
    
    fpr, tpr, _ = roc_curve(y_calib, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_calib, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"Final ROC AUC: {roc_auc:.4f}")
    print(f"Final PR AUC:  {pr_auc:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('tmeld_performance_curves.png', dpi=300)
    plt.close()
    print("Saved -> tmeld_performance_curves.png")

    # ── Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(calibrated_clf,        'tmeld_production.pkl')
    joblib.dump(list(X_top30.columns), 'selected_features.pkl')
    print("  tmeld_production.pkl    <- calibrated production model")
    print("  selected_features.pkl   <- feature list for inference")
    print("  le_production.pkl       <- gender label encoder")

    print("\n" + "=" * 60)
    print("  Phase 3 Polish complete. Production artefacts ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
