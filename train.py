import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
import optuna

warnings.filterwarnings('ignore')

def main():
    print("Loading datasets...")
    df_features = pd.read_csv('./Ds/labs_features.csv')
    df_targets = pd.read_csv('./Ds/patient_targets.csv')

    target_tests = [
        "Bilirubin, Total", "INR(PT)", "Creatinine", 
        "Platelet Count", "Alanine Aminotransferase (ALT)", 
        "Asparate Aminotransferase (AST)"
    ]

    print("Formatting for tsfresh...")
    labs_filtered = df_features[df_features['lab_test_name'].isin(target_tests)].copy()
    
    labs_filtered['charttime'] = pd.to_datetime(labs_filtered['charttime'])
    labs_filtered = labs_filtered.dropna(subset=['valuenum'])

    print("Extracting features using EfficientFCParameters...")
    extraction_settings = EfficientFCParameters()
    X_extracted = extract_features(
        labs_filtered, 
        column_id='subject_id', 
        column_sort='charttime',
        column_kind='lab_test_name',
        column_value='valuenum',
        default_fc_parameters=extraction_settings,
        impute_function=impute
    )

    print("Merging with targets...")
    merged = df_targets.merge(X_extracted, left_on='subject_id', right_index=True, how='inner')

    le = LabelEncoder()
    if merged['gender'].isnull().any():
        merged['gender'] = merged['gender'].fillna(merged['gender'].mode()[0])
    merged['gender'] = le.fit_transform(merged['gender'].astype(str))
    
    joblib.dump(le, 'le_production.pkl')

    X = merged.drop(columns=['subject_id', 'mortality_target'])
    y = merged['mortality_target'].astype(int)

    print("Rigorous feature selection via tsfresh...")
    X_selected = select_features(X, y)
    
    print("Aggressive Pruning: Keeping Top 30 features...")
    baseline_xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    baseline_xgb.fit(X_selected, y)
    
    importances = baseline_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_30_indices = indices[:30]
    top_30_features = X_selected.columns[top_30_indices]
    
    X_top30 = X_selected[top_30_features]
    print(f"Features aggressively pruned from {X_selected.shape[1]} to exactly {len(top_30_features)}.")

    # Split for calibration (Hold out 20% to calibrate probabilities)
    X_train, X_calib, y_train, y_calib = train_test_split(X_top30, y, test_size=0.2, stratify=y, random_state=42)

    print("Starting Optuna Hyperparameter Optimization...")
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 9)
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            smote_cv = SMOTE(random_state=42)
            X_train_sm_cv, y_train_sm_cv = smote_cv.fit_resample(X_train_cv, y_train_cv)
            
            model_cv = XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                subsample=subsample,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            model_cv.fit(X_train_sm_cv, y_train_sm_cv)
            preds_proba = model_cv.predict_proba(X_test_cv)[:, 1]
            scores.append(roc_auc_score(y_test_cv, preds_proba))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    print(f"Best Optuna ROC AUC: {study.best_value:.4f}")
    best_params = study.best_params

    print("Training best Pipeline and Calibrating probabilities...")
    final_smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = final_smote.fit_resample(X_train, y_train)

    final_xgb = XGBClassifier(
        **best_params,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    final_xgb.fit(X_train_sm, y_train_sm)
    
    # Probability Calibration
    calibrated_clf = CalibratedClassifierCV(estimator=final_xgb, method='isotonic', cv='prefit')
    calibrated_clf.fit(X_calib, y_calib)

    print("Exporting production artifacts...")
    joblib.dump(calibrated_clf, 'tmeld_production.pkl')
    joblib.dump(list(top_30_features), 'selected_features.pkl')

    print("Phase 3 Polish complete.")

if __name__ == "__main__":
    main()
