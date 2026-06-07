"""
HepSense -- Modular CDSS Integrator (app.py)
============================================
The "Chief Medical Officer" script. Integrates three independent expert modules
at the DECISION level (not feature level) to avoid Frankenstein data fusion.

Experts:
    1. Vision Expert   -- DANN EfficientNet-B0 (PyTorch)  -> Fibrosis stage F0-F4
    2. Clinical Expert -- XGBoost (hepsense_clinical_xgboost_v1.joblib) -> Decompensation risk
    3. Temporal Expert -- XGBoost (hepsense_temporal_xgboost_v1.joblib) -> Trajectory forecast

Final output: Rule-Based Integration Engine combining all expert opinions into
a unified HepSense Combined Recommendation.

Designed for easy Streamlit frontend integration.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# PyTorch imports for Vision module
import torch

# Local vision module
from HepSense_Vision import (
    HepSenseDANN, load_trained_model, predict_ultrasound,
    CLASS_NAMES, DEVICE, MODEL_SAVE_PATH, NUM_CLASSES, NUM_DOMAINS,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MODEL LOADING
# =============================================================================

def load_all_models():
    """
    Load all three expert models. Returns a dict of loaded models.
    Gracefully handles missing model files.
    """
    models = {}

    # --- Vision Expert (PyTorch DANN) ---
    if os.path.exists(MODEL_SAVE_PATH):
        models["vision"] = load_trained_model(MODEL_SAVE_PATH)
        print(f"[OK] Vision Expert loaded from {MODEL_SAVE_PATH}")
    else:
        models["vision"] = None
        print(f"[✗] Vision model not found at {MODEL_SAVE_PATH}. Run HepSense_Vision.py first.")

    # --- Clinical Expert (XGBoost) ---
    clin_path = "hepsense_clinical_xgboost_v1.joblib"
    if os.path.exists(clin_path):
        models["clinical"] = joblib.load(clin_path)
        print(f"[OK] Clinical Expert loaded from {clin_path}")
    else:
        models["clinical"] = None
        print(f"[✗] Clinical model not found at {clin_path}")

    # --- Temporal Expert (XGBoost) ---
    temp_path = "hepsense_temporal_xgboost_v1.joblib"
    if os.path.exists(temp_path):
        models["temporal"] = joblib.load(temp_path)
        print(f"[OK] Temporal Expert loaded from {temp_path}")
    else:
        models["temporal"] = None
        print(f"[✗] Temporal model not found at {temp_path}")

    return models


# =============================================================================
# 2. INDIVIDUAL EXPERT INFERENCE FUNCTIONS
# =============================================================================

def run_vision_expert(image_path: str, vision_model) -> dict:
    """
    Run the DANN vision model on an ultrasound image.
    Returns fibrosis stage prediction + Grad-CAM overlay.
    """
    if vision_model is None:
        return {"stage": "UNAVAILABLE", "confidence": 0.0,
                "probabilities": {}, "gradcam_overlay": None,
                "error": "Vision model not loaded."}

    result = predict_ultrasound(image_path, vision_model)
    return {
        "stage":           result["predicted_stage"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"],
        "gradcam_overlay": result["gradcam_overlay"],
        "error":           None,
    }


def run_clinical_expert(blood_data: dict, clinical_model) -> dict:
    """
    Run the clinical XGBoost model on structured blood panel data.

    Args:
        blood_data: dict matching the feature columns expected by the model.
                    E.g. from Ds/labs_features.csv format.
        clinical_model: trained XGBoost or CalibratedClassifierCV

    Returns:
        dict with risk_probability, risk_label, contributing_features
    """
    if clinical_model is None:
        return {"risk_probability": 0.0, "risk_label": "UNAVAILABLE",
                "error": "Clinical model not loaded."}

    df = pd.DataFrame([blood_data])

    # Get expected features and align
    if hasattr(clinical_model, "feature_names_in_"):
        expected = list(clinical_model.feature_names_in_)
    elif hasattr(clinical_model, "get_booster"):
        expected = clinical_model.get_booster().feature_names
    else:
        expected = list(df.columns)

    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected]

    prob = clinical_model.predict_proba(df)[0]
    risk_idx = int(np.argmax(prob))
    risk_prob = float(prob[risk_idx]) if prob.ndim == 1 else float(prob[1]) if prob.shape[1] == 2 else float(prob[risk_idx])

    # For binary: use positive class probability (threshold adjusted for 2.5% base rate)
    if len(prob) == 2:
        risk_prob = float(prob[1])
        risk_label = "HIGH" if risk_prob >= 0.05 else ("MODERATE" if risk_prob >= 0.025 else "LOW")
    else:
        risk_label = f"CLASS_{risk_idx}"

    return {"risk_probability": risk_prob, "risk_label": risk_label, "error": None}


def run_temporal_expert(blood_data: dict, temporal_model) -> dict:
    """
    Run the temporal XGBoost model for trajectory-based risk forecast.
    Uses velocity/trend features from longitudinal blood labs.
    """
    if temporal_model is None:
        return {"trend_risk": 0.0, "trend_label": "UNAVAILABLE",
                "failing_point": "Unknown", "error": "Temporal model not loaded."}

    df = pd.DataFrame([blood_data])

    if hasattr(temporal_model, "feature_names_in_"):
        expected = list(temporal_model.feature_names_in_)
    elif hasattr(temporal_model, "get_booster"):
        expected = temporal_model.get_booster().feature_names
    else:
        expected = list(df.columns)

    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected]

    prob = temporal_model.predict_proba(df)[0]
    if len(prob) == 2:
        trend_risk = float(prob[1])
    else:
        trend_risk = float(np.max(prob))

    # Identify the top failing biomarker from feature importances
    failing_point = "General Decline"
    if hasattr(temporal_model, "feature_importances_"):
        importances = temporal_model.feature_importances_
        top_idx = int(np.argmax(importances))
        if top_idx < len(expected):
            failing_point = expected[top_idx]

    trend_label = "DECLINING" if trend_risk >= 0.05 else ("WATCHFUL" if trend_risk >= 0.025 else "STABLE")

    return {"trend_risk": trend_risk, "trend_label": trend_label,
            "failing_point": failing_point, "error": None}


# =============================================================================
# 3. RULE-BASED INTEGRATION ENGINE
# =============================================================================

def integration_engine(vision_result: dict, clinical_result: dict,
                       temporal_result: dict) -> dict:
    """
    Rule-Based Integration Engine -- the "Chief Medical Officer" logic.

    Takes independent expert outputs and produces a unified HepSense
    Combined Recommendation. This is DECISION-LEVEL fusion, not feature-level.

    Returns:
        dict with severity_level, recommendation, actions, color_code
    """
    stage = vision_result.get("stage", "UNAVAILABLE")
    stage_conf = vision_result.get("confidence", 0.0)
    risk_label = clinical_result.get("risk_label", "UNAVAILABLE")
    risk_prob = clinical_result.get("risk_probability", 0.0)
    trend_label = temporal_result.get("trend_label", "UNAVAILABLE")
    trend_risk = temporal_result.get("trend_risk", 0.0)
    failing_point = temporal_result.get("failing_point", "Unknown")

    # --- ANOMALY: Acute Liver Failure (ALF) Suspected ---
    # F0 (healthy liver) but skyrocketing lab risk
    if stage == "F0" and (risk_label == "HIGH" or trend_label == "DECLINING"):
        return {
            "severity_level": "CRITICAL",
            "recommendation": (
                f"SEVERE DISCORDANCE (ALF SUSPECTED): Vision shows healthy liver "
                f"({stage}, {stage_conf:.0%}), but clinical risk is {risk_label} "
                f"({risk_prob:.0%}) with {trend_label} trajectory. "
                f"Primary anomaly: {failing_point}. "
                "WARNING: Rule out Acute Liver Failure (e.g., acetaminophen toxicity, "
                "acute viral hepatitis, ischemic hepatopathy). Immediate STAT labs required."
            ),
            "actions": [
                "STAT liver function panel and coagulation profile",
                "Rule out acetaminophen toxicity (NAC protocol)",
                "Immediate inpatient admission (ICU monitoring)",
                "Urgent hepatology consultation",
            ],
            "color_code": "red",
        }

    # --- CRITICAL: F3/F4 with high clinical risk ---
    if stage in ("F4",) and risk_label == "HIGH":
        return {
            "severity_level": "CRITICAL",
            "recommendation": (
                f"CRITICAL: F4 Cirrhosis detected ({stage_conf:.0%} confidence) "
                f"with HIGH decompensation risk ({risk_prob:.0%}). "
                f"Primary failing biomarker: {failing_point}. "
                "Fast-track transplant evaluation. Initiate portal hypertension "
                "screening (EGD for varices). Hepatology consult STAT."
            ),
            "actions": [
                "Immediate hepatology referral",
                "Schedule EGD for variceal screening",
                "Initiate transplant evaluation workup",
                "ICU standby for acute decompensation",
                "Start prophylactic beta-blocker therapy",
            ],
            "color_code": "red",
        }

    if stage in ("F3", "F4") and trend_label == "DECLINING":
        return {
            "severity_level": "CRITICAL",
            "recommendation": (
                f"CRITICAL: Advanced fibrosis ({stage}, {stage_conf:.0%}) with "
                f"DECLINING trajectory (trend risk {trend_risk:.0%}). "
                f"Key deterioration driver: {failing_point}. "
                "Urgent inpatient admission recommended. High risk of "
                "hepatic encephalopathy or variceal hemorrhage."
            ),
            "actions": [
                "Urgent inpatient admission",
                "Serial blood panel monitoring (q6h)",
                "Lactulose/rifaximin for encephalopathy prevention",
                "Hepatology and transplant team consultation",
            ],
            "color_code": "red",
        }

    if stage == "F3" and risk_label in ("HIGH", "MODERATE"):
        return {
            "severity_level": "HIGH",
            "recommendation": (
                f"HIGH RISK: F3 fibrosis ({stage_conf:.0%}) with {risk_label} "
                f"decompensation risk ({risk_prob:.0%}). "
                f"Monitor {failing_point} closely. "
                "Aggressive antifibrotic therapy and close follow-up required."
            ),
            "actions": [
                "Hepatology referral within 1 week",
                "FibroScan / elastography for confirmation",
                "Repeat blood panel in 7 days",
                "Evaluate for antifibrotic therapy",
            ],
            "color_code": "orange",
        }

    # --- MODERATE: Early-to-mid fibrosis with warning signs ---
    if stage in ("F1", "F2") and (risk_label == "HIGH" or trend_label == "DECLINING"):
        return {
            "severity_level": "MODERATE",
            "recommendation": (
                f"MODERATE RISK: Early fibrosis ({stage}, {stage_conf:.0%}) but "
                f"clinical trajectory is concerning -- {risk_label} risk "
                f"({risk_prob:.0%}), trajectory: {trend_label}. "
                f"Biomarker of concern: {failing_point}. "
                "Lifestyle intervention + close surveillance needed."
            ),
            "actions": [
                "Schedule hepatology follow-up in 2 weeks",
                "Lifestyle modification counseling (alcohol cessation, diet)",
                "Repeat labs in 14 days to confirm trajectory",
                "Consider liver biopsy for staging confirmation",
            ],
            "color_code": "yellow",
        }

    if stage in ("F1", "F2") and risk_label == "MODERATE":
        return {
            "severity_level": "LOW-MODERATE",
            "recommendation": (
                f"LOW-MODERATE: {stage} fibrosis ({stage_conf:.0%}), "
                f"moderate clinical risk ({risk_prob:.0%}). "
                "Continue current management with periodic monitoring."
            ),
            "actions": [
                "Routine follow-up in 3-6 months",
                "Annual FibroScan monitoring",
                "Lifestyle modification counseling",
            ],
            "color_code": "yellow",
        }

    # --- LOW: F0 or minimal risk ---
    if stage == "F0":
        return {
            "severity_level": "LOW",
            "recommendation": (
                f"LOW RISK: No significant fibrosis detected ({stage}, "
                f"{stage_conf:.0%}). Clinical risk: {risk_label} ({risk_prob:.0%}). "
                "Routine surveillance recommended."
            ),
            "actions": [
                "Annual liver function panel",
                "Lifestyle counseling (hepatoprotective diet)",
                "Re-image in 12-24 months if risk factors persist",
            ],
            "color_code": "green",
        }

    # --- DEFAULT FALLBACK ---
    return {
        "severity_level": "INDETERMINATE",
        "recommendation": (
            f"Stage: {stage} ({stage_conf:.0%}), "
            f"Clinical risk: {risk_label} ({risk_prob:.0%}), "
            f"Trajectory: {trend_label} ({trend_risk:.0%}). "
            "Insufficient data for a definitive recommendation. "
            "Recommend hepatology consultation for comprehensive evaluation."
        ),
        "actions": ["Hepatology consultation", "Complete blood panel", "Imaging follow-up"],
        "color_code": "gray",
    }


# =============================================================================
# 4. MAIN DASHBOARD FUNCTION (Streamlit-Ready)
# =============================================================================

def hep_sense_clinical_dashboard(patient_image: str,
                                  patient_blood_data: dict,
                                  models: dict = None) -> dict:
    """
    The primary entry point for the HepSense Modular CDSS.

    Args:
        patient_image:      Path to the patient's ultrasound image.
        patient_blood_data: Dict of blood panel values matching model features.
        models:             Pre-loaded model dict (from load_all_models()).

    Returns:
        Comprehensive dict with:
            - vision_result:   Stage prediction + Grad-CAM
            - clinical_result: Decompensation risk
            - temporal_result: Trajectory risk + failing point
            - recommendation:  Unified HepSense Combined Recommendation
    """
    if models is None:
        models = load_all_models()

    print("\n" + "=" * 60)
    print("  HepSense Modular CDSS -- Running All Expert Modules")
    print("=" * 60)

    # --- Expert 1: Vision ---
    print("\n[1/3] Vision Expert (DANN EfficientNet-B0)...")
    vision_result = run_vision_expert(patient_image, models.get("vision"))
    print(f"      -> Stage: {vision_result['stage']} "
          f"({vision_result['confidence']:.1%})")

    # --- Expert 2: Clinical ---
    print("[2/3] Clinical Expert (XGBoost)...")
    clinical_result = run_clinical_expert(patient_blood_data, models.get("clinical"))
    print(f"      -> Risk: {clinical_result['risk_label']} "
          f"({clinical_result['risk_probability']:.1%})")

    # --- Expert 3: Temporal ---
    print("[3/3] Temporal Expert (XGBoost)...")
    temporal_result = run_temporal_expert(patient_blood_data, models.get("temporal"))
    print(f"      -> Trend: {temporal_result['trend_label']} "
          f"({temporal_result['trend_risk']:.1%})")

    # --- Integration Engine ---
    print("\n[CMO] Rule-Based Integration Engine...")
    recommendation = integration_engine(vision_result, clinical_result, temporal_result)
    print(f"      -> Severity: {recommendation['severity_level']}")
    print(f"      -> {recommendation['recommendation']}")

    return {
        "vision_result":   vision_result,
        "clinical_result": clinical_result,
        "temporal_result": temporal_result,
        "recommendation":  recommendation,
    }


# =============================================================================
# 5. CLI DEMO
# =============================================================================

if __name__ == "__main__":
    print("HepSense Modular CDSS -- CLI Demo")
    print("-" * 40)

    all_models = load_all_models()

    # Demo blood data matching clinical_pipeline.py features
    demo_blood = {
        "Bili_last": 3.2, "Bili_max": 4.8, "Bili_velocity": 1.6,
        "Creat_last": 1.8, "Creat_max": 2.1, "Creat_velocity": 0.5,
        "INR_last": 1.9, "INR_max": 2.3, "INR_velocity": 0.4,
        "Sodium_last": 131.0, "Sodium_min": 128.0,
        "Platelets_last": 78.0,
        "NLP_Encephalopathy_Flag": 1,
        "NLP_Variceal_Bleeding_Flag": 0,
    }

    # Use first F4 image as demo if available
    demo_img = None
    f4_dir = os.path.join("Dataset", "F4")
    if os.path.isdir(f4_dir):
        imgs = [f for f in os.listdir(f4_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if imgs:
            demo_img = os.path.join(f4_dir, imgs[0])

    if demo_img:
        result = hep_sense_clinical_dashboard(demo_img, demo_blood, all_models)

        print("\n" + "=" * 60)
        print("  FINAL HEPSENSE COMBINED RECOMMENDATION")
        print("=" * 60)
        rec = result["recommendation"]
        print(f"  Severity : {rec['severity_level']}")
        print(f"  Summary  : {rec['recommendation']}")
        print(f"  Actions  :")
        for a in rec["actions"]:
            print(f"    • {a}")

        # Save Grad-CAM if available
        cam = result["vision_result"].get("gradcam_overlay")
        if cam is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(cam)
            plt.title(f"Grad-CAM -- {result['vision_result']['stage']}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("hepsense_gradcam_output.png", dpi=150)
            plt.close()
            print("\n  Grad-CAM saved -> hepsense_gradcam_output.png")
    else:
        print("[WARN] No demo image found in Dataset/F4. Skipping vision demo.")
        print("       Provide an image path to hep_sense_clinical_dashboard().")
