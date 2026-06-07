import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
labels = pd.read_csv('decompensation_labels.csv')
if 'subject_id' in labels.columns: labels.rename(columns={'subject_id': 'patient_id'}, inplace=True)
cohort = pd.read_csv('cirrhosis_cohort.csv')
if 'subject_id' in cohort.columns: cohort.rename(columns={'subject_id': 'patient_id'}, inplace=True)
labs = pd.read_csv('labs_cirrhosis.csv')
if 'subject_id' in labs.columns: labs.rename(columns={'subject_id': 'patient_id'}, inplace=True)

print("Computing Baseline MELD...")
labs['charttime'] = pd.to_datetime(labs['charttime'])
labs = labs.sort_values(['patient_id', 'charttime'])
last_obs = labs.groupby('patient_id').last()

def row_meld(row):
    cr = row.get('Creatinine', 1.0)
    tb = row.get('Bilirubin, Total', 1.0)
    inr = row.get('INR(PT)', 1.0)
    if pd.isna(cr): cr = 1.0
    if pd.isna(tb): tb = 1.0
    if pd.isna(inr): inr = 1.0
    cr, tb, inr = max(cr, 1.0), max(tb, 1.0), max(inr, 1.0)
    score = 3.78 * np.log(tb) + 11.2 * np.log(inr) + 9.57 * np.log(cr) + 6.43
    return np.clip(score, 6, 40)

last_obs['MELD_Score'] = last_obs.apply(row_meld, axis=1)
clinical_scores = last_obs[['MELD_Score']]

# Load existing extracted features to get the same exact row alignment
X_extracted = pd.read_parquet('tsfresh_cache.parquet')
X_extracted = X_extracted.merge(clinical_scores, left_index=True, right_index=True, how='left')

# Add NLP
notes = pd.read_csv('discharge_notes.csv')
if 'subject_id' in notes.columns:
    notes.rename(columns={'subject_id': 'patient_id'}, inplace=True)
notes['NLP_Encephalopathy_Flag'] = notes['text'].str.contains('encephalopathy|confusion|asterixis', case=False, na=False).astype(int)
notes['NLP_Variceal_Bleeding_Flag'] = notes['text'].str.contains('varices|variceal bleed|banding', case=False, na=False).astype(int)
nlp_flags = notes.groupby('patient_id')[['NLP_Encephalopathy_Flag', 'NLP_Variceal_Bleeding_Flag']].max()
X_extracted = X_extracted.merge(nlp_flags, left_index=True, right_index=True, how='left')

merged = pd.merge(X_extracted.reset_index().rename(columns={'index': 'patient_id', 'id': 'patient_id'}), labels, on='patient_id', how='inner')
merged = pd.merge(merged, cohort[['patient_id', 'age', 'gender']], on='patient_id', how='left')
merged.rename(columns={'gender': 'sex'}, inplace=True)

le = joblib.load('le_production.pkl')
merged['sex'] = le.transform(merged['sex'])

y = merged['decompensation_90day'].astype(int)
X = merged.drop(columns=['decompensation_90day'])

medians = X.median(numeric_only=True)
X = X.fillna(medians)

selected_features = joblib.load('selected_features.pkl')
X_final = X[selected_features]

X_train, X_calib, y_train, y_calib = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)
X_train_meld, X_calib_meld, _, _ = train_test_split(X[['MELD_Score']], y, test_size=0.2, stratify=y, random_state=42)

meld_auroc = roc_auc_score(y_calib, X_calib_meld['MELD_Score'])
print(f"\n--- BASELINE MELD AUROC (on calibration set): {meld_auroc:.4f} ---")

model = joblib.load('tmeld_production.pkl')
y_pred_proba = model.predict_proba(X_calib)[:, 1]

model_auroc = roc_auc_score(y_calib, y_pred_proba)
print(f"--- T-MELD AUROC (on calibration set): {model_auroc:.4f} ---")

print("\n--- THRESHOLD ANALYSIS FOR T-MELD ---")
print("Target: Boost Recall > 70%")
thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
for t in thresholds:
    preds = (y_pred_proba >= t).astype(int)
    cm = confusion_matrix(y_calib, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        continue # all one class predicted
    recall = tp / (tp + fn) if (tp+fn)>0 else 0
    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    print(f"Threshold: {t:5.3f} | Recall: {recall:4.2f} | Specificity: {spec:4.2f} | Precision: {precision:4.2f} | FP: {fp:3d} | FN: {fn:3d} | TP: {tp:3d}")
