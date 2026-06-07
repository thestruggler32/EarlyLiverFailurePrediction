import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

labels = pd.read_csv('decompensation_labels.csv')
if 'subject_id' in labels.columns: labels.rename(columns={'subject_id': 'patient_id'}, inplace=True)

labs = pd.read_csv('labs_cirrhosis.csv')
if 'subject_id' in labs.columns: labs.rename(columns={'subject_id': 'patient_id'}, inplace=True)
labs['charttime'] = pd.to_datetime(labs['charttime'])
labs_pivot = labs.pivot_table(index=['patient_id', 'charttime'], columns='lab_test_name', values='valuenum', aggfunc='mean').reset_index()
labs_pivot = labs_pivot.sort_values(['patient_id', 'charttime'])
last_obs = labs_pivot.groupby('patient_id').last()

def row_meld(row):
    cr = row.get('Creatinine', 1.0)
    tb = row.get('Bilirubin, Total', 1.0)
    inr = row.get('INR(PT)', 1.0)
    cr = max(cr if not pd.isna(cr) else 1.0, 1.0)
    tb = max(tb if not pd.isna(tb) else 1.0, 1.0)
    inr = max(inr if not pd.isna(inr) else 1.0, 1.0)
    score = 3.78 * np.log(tb) + 11.2 * np.log(inr) + 9.57 * np.log(cr) + 6.43
    return np.clip(score, 6, 40)

last_obs['MELD_Score'] = last_obs.apply(row_meld, axis=1)
merged = pd.merge(last_obs[['MELD_Score']].reset_index(), labels, on='patient_id', how='inner')

auroc = roc_auc_score(merged['decompensation_90day'], merged['MELD_Score'])
print(f"BASELINE MELD AUROC: {auroc:.4f}")
