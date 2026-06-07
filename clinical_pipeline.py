import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import joblib

print("=======================================================")
print("CLINICAL PIPELINE — Enhanced with MELD, FIB-4, Real NLP Flags")
print("=======================================================")

np.random.seed(42)
n_patients = 2000
days = 14

print(f"\n[1/7] Generating synthetic {days}-day longitudinal clinical histories "
      f"for {n_patients} patients...")

records = []
for pid in range(n_patients):
    # Patient base severity — lognormal to mimic real lab distributions
    base_bili  = np.random.lognormal(mean=0.5, sigma=0.8)
    base_creat = np.random.lognormal(mean=0.2, sigma=0.5)
    base_inr   = np.random.normal(loc=1.2, scale=0.3)
    base_alt   = np.random.lognormal(mean=3.5, sigma=0.6)   # ~35 IU/L baseline
    base_ast   = np.random.lognormal(mean=3.5, sigma=0.6)
    age        = np.random.randint(35, 80)

    declining = np.random.choice([True, False], p=[0.3, 0.7])

    for day in range(1, days + 1):
        trend = (day / days) if declining else 0

        bili     = max(0.1,  base_bili  + trend * np.random.normal(2.0, 0.5) + np.random.normal(0, 0.10))
        creat    = max(0.1,  base_creat + trend * np.random.normal(1.0, 0.2) + np.random.normal(0, 0.05))
        inr      = max(0.8,  base_inr   + trend * np.random.normal(0.5, 0.1) + np.random.normal(0, 0.05))
        sodium   = np.clip(np.random.normal(138, 4) - trend * 5, 110, 160)
        platelets = max(10,  np.random.normal(120, 40) - trend * 30)
        alt      = max(5,    base_alt   + trend * np.random.normal(20, 5)  + np.random.normal(0, 2))
        ast      = max(5,    base_ast   + trend * np.random.normal(25, 6)  + np.random.normal(0, 2))

        records.append({
            'patient_id':      pid,
            'day':             day,
            'age':             age,
            'Bilirubin':       bili,
            'Creatinine':      creat,
            'INR':             inr,
            'Sodium':          sodium,
            'Platelets':       platelets,
            'ALT':             alt,
            'AST':             ast,
            'declining_status': declining,
        })

df_long = pd.DataFrame(records)


print("\n[2/7] Extracting temporal features "
      "(last, max, velocity, 7-day rolling velocity, acceleration)...")

def extract_features(group):
    first = group.iloc[0]
    last  = group.iloc[-1]

    # ── Basic: last value, maximum, first-to-last velocity ────────────────────
    feats = {
        'Bili_last':     last['Bilirubin'],
        'Bili_max':      group['Bilirubin'].max(),
        'Bili_velocity': last['Bilirubin'] - first['Bilirubin'],

        'Creat_last':     last['Creatinine'],
        'Creat_max':      group['Creatinine'].max(),
        'Creat_velocity': last['Creatinine'] - first['Creatinine'],

        'INR_last':     last['INR'],
        'INR_max':      group['INR'].max(),
        'INR_velocity': last['INR'] - first['INR'],

        'Sodium_last': last['Sodium'],
        'Sodium_min':  group['Sodium'].min(),

        'Platelets_last': last['Platelets'],
        'Platelets_min':  group['Platelets'].min(),

        'ALT_last': last['ALT'],
        'ALT_max':  group['ALT'].max(),
        'AST_last': last['AST'],
        'AST_max':  group['AST'].max(),

        'age': last['age'],
    }

    # ── Rolling 7-day velocity: slope over the LAST 7 days only ───────────────
    # Captures recent acceleration, not just overall trend
    last7 = group.tail(7)
    first7 = last7.iloc[0]
    feats['Bili_velocity_7d']  = last7['Bilirubin'].iloc[-1]  - first7['Bilirubin']
    feats['Creat_velocity_7d'] = last7['Creatinine'].iloc[-1] - first7['Creatinine']
    feats['INR_velocity_7d']   = last7['INR'].iloc[-1]        - first7['INR']

    # ── Acceleration: how much did the rate of change itself change? ──────────
    # Split 14 days into first half (1-7) and second half (8-14)
    mid = group.iloc[6]
    first_half_slope = mid['Bilirubin'] - first['Bilirubin']
    second_half_slope = last['Bilirubin'] - mid['Bilirubin']
    feats['Bili_acceleration'] = second_half_slope - first_half_slope

    first_half_creat = mid['Creatinine'] - first['Creatinine']
    second_half_creat = last['Creatinine'] - mid['Creatinine']
    feats['Creat_acceleration'] = second_half_creat - first_half_creat

    # ── Target ────────────────────────────────────────────────────────────────
    feats['Target_14Day_Decompensation'] = int(last['declining_status'])

    return pd.Series(feats)

df_features = (
    df_long.groupby('patient_id')
    .apply(extract_features, include_groups=False)
    .reset_index()
)


print("\n[3/7] Computing clinical scores (MELD + FIB-4) as model features...")

# ── MELD Score ─────────────────────────────────────────────────────────────
# Formula: 3.78×ln(bili) + 11.2×ln(INR) + 9.57×ln(creat) + 6.43
# Clinically validated mortality predictor — we give it to the AI as a feature,
# NOT just as a benchmark to beat.
def calc_meld(row):
    b = max(1.0, row['Bili_last'])
    c = min(4.0, max(1.0, row['Creat_last']))
    i = max(1.0, row['INR_last'])
    return 3.78 * np.log(b) + 11.2 * np.log(i) + 9.57 * np.log(c) + 6.43

df_features['MELD_Score'] = df_features.apply(calc_meld, axis=1)

# ── FIB-4 Index ────────────────────────────────────────────────────────────
# Formula: (age × AST) / (platelets × √ALT)
# Validated non-invasive fibrosis marker. >3.25 = advanced fibrosis.
def calc_fib4(row):
    denom = row['Platelets_last'] * np.sqrt(max(1.0, row['ALT_last']))
    if denom <= 0:
        return np.nan
    return (row['age'] * row['AST_last']) / denom

df_features['FIB4_Score'] = df_features.apply(calc_fib4, axis=1)
df_features['FIB4_Score'] = df_features['FIB4_Score'].clip(0, 20)  # cap outliers

print(f"  MELD Score — mean: {df_features['MELD_Score'].mean():.2f}, "
      f"max: {df_features['MELD_Score'].max():.2f}")
print(f"  FIB-4 Score — mean: {df_features['FIB4_Score'].mean():.2f}, "
      f"max: {df_features['FIB4_Score'].max():.2f}")


print("\n[4/7] Setting realistic NLP flags (correlated with lab severity)...")

# ── Encephalopathy flag ─────────────────────────────────────────────────────
# In real patients, hepatic encephalopathy occurs when the liver can no longer
# filter toxins. High bilirubin + rising INR are the strongest lab predictors.
# Probability: base 5%, +20% if bilirubin > 3, +15% if INR > 1.8
enc_prob = (
    0.05
    + (df_features['Bili_last'] > 3.0).astype(float) * 0.20
    + (df_features['INR_last'] > 1.8).astype(float) * 0.15
    + (df_features['Bili_acceleration'] > 1.0).astype(float) * 0.10
).clip(0, 0.90)

df_features['NLP_Encephalopathy_Flag'] = np.random.binomial(
    n=1, p=enc_prob.values
)

# ── Variceal bleeding flag ──────────────────────────────────────────────────
# Varices form from portal hypertension. Low platelets + low sodium are proxies.
# Probability: base 3%, +15% if platelets < 80, +10% if sodium < 132
var_prob = (
    0.03
    + (df_features['Platelets_last'] < 80).astype(float) * 0.15
    + (df_features['Sodium_last'] < 132).astype(float) * 0.10
    + (df_features['INR_velocity'] > 0.3).astype(float) * 0.07
).clip(0, 0.80)

df_features['NLP_Variceal_Bleeding_Flag'] = np.random.binomial(
    n=1, p=var_prob.values
)

# Update target to include NLP-flagged events (they indicate decompensation)
df_features['Target_14Day_Decompensation'] = np.clip(
    df_features['Target_14Day_Decompensation']
    + df_features['NLP_Encephalopathy_Flag']
    + df_features['NLP_Variceal_Bleeding_Flag'],
    0, 1
)

pos_rate = df_features['Target_14Day_Decompensation'].mean()
print(f"  Positive rate after NLP: {pos_rate:.1%} "
      f"({df_features['Target_14Day_Decompensation'].sum()} / {n_patients})")


print("\n[5/7] Splitting into train / test cohorts...")

feature_cols = [
    # Bilirubin
    'Bili_last', 'Bili_max', 'Bili_velocity', 'Bili_velocity_7d', 'Bili_acceleration',
    # Creatinine
    'Creat_last', 'Creat_max', 'Creat_velocity', 'Creat_velocity_7d', 'Creat_acceleration',
    # INR
    'INR_last', 'INR_max', 'INR_velocity', 'INR_velocity_7d',
    # Sodium
    'Sodium_last', 'Sodium_min',
    # Platelets
    'Platelets_last', 'Platelets_min',
    # Liver enzymes
    'ALT_last', 'ALT_max', 'AST_last', 'AST_max',
    # Demographics
    'age',
    # NLP flags
    'NLP_Encephalopathy_Flag', 'NLP_Variceal_Bleeding_Flag',
    # *** Clinical scores — NOW included as features, not just benchmarks ***
    'MELD_Score', 'FIB4_Score',
]

X = df_features[feature_cols]
y = df_features['Target_14Day_Decompensation']
meld_baseline = df_features['MELD_Score']

X_train, X_test, y_train, y_test, meld_train, meld_test = train_test_split(
    X, y, meld_baseline, test_size=0.2, random_state=42, stratify=y
)


print("\n[6/7] Training Enhanced Temporal XGBoost Model...")

# Handle class imbalance: if 30% of patients decline, weight the minority class
# by the inverse ratio so the model takes HIGH-RISK cases seriously
pos  = y_train.sum()
neg  = len(y_train) - pos
spw  = neg / pos   # scale_pos_weight: tells XGBoost to penalise missing positives more
print(f"  Class balance — positive: {pos}, negative: {neg}, scale_pos_weight: {spw:.1f}")

model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,   # ← new: corrects imbalance
    random_state=42,
    eval_metric='aucpr',    # ← new: AUCPR is better than AUROC for imbalanced problems
)

model.fit(X_train, y_train)

print("\n[7/7] BENCHMARK SHOWDOWN + SHAP explainability...")
y_pred_proba  = model.predict_proba(X_test)[:, 1]
meld_auc      = roc_auc_score(y_test, meld_test)
ai_auc        = roc_auc_score(y_test, y_pred_proba)

print("-" * 55)
print(f"[Hospital] Static MELD Score AUROC : {meld_auc:.4f}")
print(f"[AI]       HepSense Temporal AUROC : {ai_auc:.4f}")
print(f"[Gain]     Delta AUROC             : +{ai_auc - meld_auc:.4f}")
print("-" * 55)

if ai_auc > meld_auc:
    print("SUCCESS: Temporal AI outperforms the static MELD baseline!")

# ── SHAP ───────────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance (Enhanced Pipeline — MELD + FIB-4 + Velocity)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved -> shap_summary_plot.png")

joblib.dump(model, "hepsense_temporal_xgboost_v1.joblib")
print("Model saved -> hepsense_temporal_xgboost_v1.joblib")

print("\n=======================================================")
print("PIPELINE COMPLETE.")
print("=======================================================")
