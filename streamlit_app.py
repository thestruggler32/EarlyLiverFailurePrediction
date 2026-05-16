"""
HepSense -- Streamlit Clinical Dashboard (v2)
===============================================
Two inputs in the main area:
  LEFT  : Upload ultrasound image  -> DANN Vision Expert (F-stage + Grad-CAM)
  RIGHT : Upload EHR CSV           -> t-MELD Temporal Expert (calibrated risk)

Both feed into the Rule-Based Integration Engine for a unified recommendation.

Run:  python -m streamlit run streamlit_app.py
"""

import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from PIL import Image

# -- Page config --
st.set_page_config(
    page_title="HepSense CDSS",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- HepSense modules --
from HepSense_Vision import (
    HepSenseDANN, load_trained_model, predict_ultrasound,
    CLASS_NAMES, DEVICE, MODEL_SAVE_PATH,
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%);
        padding: 40px 30px; border-radius: 16px; margin-bottom: 30px;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 42px; font-weight: 800; margin: 0; letter-spacing: -1px; }
    .main-header p  { font-size: 16px; opacity: 0.85; margin-top: 8px; }

    .severity-card {
        padding: 30px; border-radius: 16px; text-align: center;
        margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .severity-CRITICAL       { background: linear-gradient(135deg,#7f1d1d,#991b1b); color:#fecaca; border:2px solid #f87171; }
    .severity-HIGH           { background: linear-gradient(135deg,#78350f,#92400e); color:#fde68a; border:2px solid #fbbf24; }
    .severity-MODERATE       { background: linear-gradient(135deg,#713f12,#854d0e); color:#fef3c7; border:2px solid #f59e0b; }
    .severity-LOW-MODERATE   { background: linear-gradient(135deg,#365314,#3f6212); color:#d9f99d; border:2px solid #84cc16; }
    .severity-LOW            { background: linear-gradient(135deg,#064e3b,#065f46); color:#a7f3d0; border:2px solid #34d399; }
    .severity-INDETERMINATE  { background: linear-gradient(135deg,#1e293b,#334155); color:#cbd5e1; border:2px solid #64748b; }

    .severity-label { font-size: 14px; text-transform: uppercase; letter-spacing: 3px; font-weight: 600; opacity: 0.8; }
    .severity-level { font-size: 48px; font-weight: 800; margin: 10px 0; }

    .expert-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .expert-card h3    { margin: 0 0 6px 0; font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 2px; }
    .expert-card .value { font-size: 36px; font-weight: 800; color: #0f172a; }
    .expert-card .sub   { font-size: 13px; color: #94a3b8; margin-top: 4px; }

    .rec-box {
        background: #f1f5f9; border-left: 6px solid #3b82f6;
        padding: 20px 24px; border-radius: 0 12px 12px 0;
        font-size: 16px; line-height: 1.7; color: #1e293b; margin: 16px 0;
    }
    .action-item {
        background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px 16px; margin: 6px 0; font-size: 14px; color: #334155;
    }
    .input-section {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 24px; min-height: 320px;
    }
    .input-section h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>HepSense CDSS</h1>
    <p>Multi-Modal Liver Cirrhosis Staging -- DANN Vision + t-MELD Temporal Risk Engine</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Load models (cached)
# =============================================================================
@st.cache_resource
def load_vision_model():
    if os.path.exists(MODEL_SAVE_PATH):
        return load_trained_model(MODEL_SAVE_PATH)
    return None

@st.cache_resource
def load_tmeld_artifacts():
    """Load the t-MELD production model and feature config."""
    try:
        from tsfresh.feature_extraction.settings import from_columns
        model = joblib.load('tmeld_production.pkl')
        selected_cols = joblib.load('selected_features.pkl')
        try:
            le = joblib.load('le_production.pkl')
        except FileNotFoundError:
            le = None
        static_feats = ['age', 'gender']
        ts_cols = [c for c in selected_cols if c not in static_feats]
        kind_to_fc = from_columns(ts_cols)
        return model, selected_cols, le, kind_to_fc
    except Exception as e:
        st.error(f"t-MELD artifacts missing: {e}")
        return None, None, None, None

vision_model = load_vision_model()
tmeld_model, selected_cols, le, kind_to_fc = load_tmeld_artifacts()

# =============================================================================
# INPUTS -- Two columns: Image (left) + EHR (right)
# =============================================================================
col_img, col_ehr = st.columns(2)

uploaded_image = None
with col_img:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### Upload Ultrasound Image")
    st.caption("Liver ultrasound for F-stage classification via DANN EfficientNet-B0")
    uploaded_image = st.file_uploader(
        "Drag & drop or browse", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="img_upload",
    )
    if uploaded_image:
        preview = Image.open(uploaded_image)
        st.image(preview, use_container_width=True, caption="Uploaded ultrasound")
    st.markdown('</div>', unsafe_allow_html=True)

df_ehr = None
with col_ehr:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### Upload EHR Labs (t-MELD)")
    st.caption("Longitudinal blood panel CSV for temporal risk prediction")

    # Demographics
    c1, c2 = st.columns(2)
    with c1:
        age_val = st.number_input("Patient Age", 18, 120, 55)
    with c2:
        if le is not None:
            gender_val = st.selectbox("Gender", le.classes_)
        else:
            gender_val = st.selectbox("Gender", ["M", "F"])

    tab_upload, tab_paste = st.tabs(["CSV Upload", "Quick Paste"])

    with tab_upload:
        ehr_file = st.file_uploader(
            "Upload EHR CSV",
            type=["csv"],
            help="CSV with columns: charttime, lab_test_name or test columns",
            key="ehr_upload",
        )
        if ehr_file:
            try:
                df_ehr = pd.read_csv(ehr_file)
                st.success(f"Loaded {len(df_ehr)} rows")
            except Exception as e:
                st.error(f"CSV parse error: {e}")

    with tab_paste:
        pasted = st.text_area(
            "Paste EHR data (CSV format)", height=120,
            placeholder='charttime,Bilirubin Total,INR(PT),Creatinine\n2023-10-01,1.2,1.1,0.9\n2023-10-05,2.1,1.4,1.2',
        )
        if pasted and not ehr_file:
            try:
                sep = ',' if ',' in pasted.split('\n')[0] else '\t'
                df_ehr = pd.read_csv(io.StringIO(pasted), sep=sep)
                st.success(f"Parsed {len(df_ehr)} rows")
            except Exception:
                st.error("Could not parse pasted data.")

    if df_ehr is not None:
        st.dataframe(df_ehr.head(5), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# t-MELD Inference Function
# =============================================================================
TARGET_TESTS = [
    "Bilirubin, Total", "INR(PT)", "Creatinine",
    "Platelet Count", "Alanine Aminotransferase (ALT)",
    "Asparate Aminotransferase (AST)",
]

def run_tmeld_pipeline(df_input, age, gender):
    """
    Run the full t-MELD pipeline: preprocess EHR -> tsfresh extraction -> calibrated prediction.
    Returns (probability, risk_label) or raises on error.
    """
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    # Detect format: long-form (lab_test_name column) vs wide-form (test columns)
    if 'lab_test_name' in df_input.columns:
        # Long format from labs_features.csv
        cols_lower = {c.lower(): c for c in df_input.columns}
        charttime_col = cols_lower.get('charttime', 'charttime')
        df_input = df_input.rename(columns={charttime_col: 'charttime'})
        df_input['charttime'] = pd.to_datetime(df_input['charttime'], errors='coerce')
        df_input = df_input.dropna(subset=['charttime', 'valuenum'])
        df_input = df_input[df_input['lab_test_name'].isin(TARGET_TESTS)]

        if 'subject_id' not in df_input.columns:
            df_input['subject_id'] = 1

        df_long = df_input[['subject_id', 'charttime', 'lab_test_name', 'valuenum']].copy()
    else:
        # Wide format: columns are test names
        cols_lower = {c.lower(): c for c in df_input.columns}
        charttime_col = cols_lower.get('charttime', None)
        if charttime_col is None:
            raise ValueError("CSV must contain a 'charttime' column.")
        df_input = df_input.rename(columns={cols_lower['charttime']: 'charttime'})
        df_input['charttime'] = pd.to_datetime(df_input['charttime'], errors='coerce')
        df_input = df_input.dropna(subset=['charttime'])

        available = [c for c in TARGET_TESTS if c in df_input.columns]
        if not available:
            raise ValueError(f"No target tests found. Expected: {TARGET_TESTS}")

        df_input = df_input.sort_values('charttime')
        df_input.set_index('charttime', inplace=True)
        df_daily = df_input[available].resample('D').mean().ffill()
        df_clean = df_daily.reset_index()
        df_clean['subject_id'] = 1

        df_long = pd.melt(
            df_clean, id_vars=['subject_id', 'charttime'],
            value_vars=available, var_name='lab_test_name', value_name='valuenum',
        ).dropna()

    # tsfresh extraction
    X_extracted = extract_features(
        df_long, column_id='subject_id', column_sort='charttime',
        column_kind='lab_test_name', column_value='valuenum',
        kind_to_fc_parameters=kind_to_fc,
        impute_function=impute, disable_progressbar=True,
    )

    # Demographics
    if le is not None:
        try:
            gender_enc = le.transform([gender])[0]
        except Exception:
            gender_enc = 1 if gender == "M" else 0
    else:
        gender_enc = 1 if gender == "M" else 0

    X_extracted['gender'] = gender_enc
    X_extracted['age'] = age

    for col in selected_cols:
        if col not in X_extracted.columns:
            X_extracted[col] = 0.0
    X_final = X_extracted[selected_cols]

    # Prediction
    if hasattr(tmeld_model, 'predict_proba'):
        prob = tmeld_model.predict_proba(X_final)[0][1]
    else:
        prob = float(tmeld_model.predict(X_final)[0])

    if prob < 0.20:
        label = "LOW"
    elif prob < 0.50:
        label = "MODERATE"
    else:
        label = "HIGH"

    return prob, label

# =============================================================================
# Integration Engine
# =============================================================================
def integration_engine(stage, stage_conf, risk_label, risk_prob):
    """Rule-based fusion of Vision stage + t-MELD risk."""

    if stage == "F4" and risk_label == "HIGH":
        return {
            "severity": "CRITICAL",
            "recommendation": (
                f"CRITICAL: F4 Cirrhosis ({stage_conf:.0%} confidence) + HIGH decompensation risk ({risk_prob:.0%}). "
                "Fast-track transplant evaluation. Portal hypertension screening (EGD for varices). Hepatology consult STAT."
            ),
            "actions": [
                "Immediate hepatology referral",
                "Schedule EGD for variceal screening",
                "Initiate transplant evaluation workup",
                "ICU standby for acute decompensation",
                "Start prophylactic beta-blocker therapy",
            ],
        }
    if stage in ("F3", "F4") and risk_label in ("HIGH", "MODERATE"):
        return {
            "severity": "CRITICAL" if risk_label == "HIGH" else "HIGH",
            "recommendation": (
                f"{'CRITICAL' if risk_label=='HIGH' else 'HIGH RISK'}: {stage} fibrosis ({stage_conf:.0%}) "
                f"with {risk_label} temporal decompensation risk ({risk_prob:.0%}). "
                "Aggressive monitoring and antifibrotic therapy required."
            ),
            "actions": [
                "Urgent hepatology referral",
                "Serial blood panel monitoring",
                "FibroScan / elastography confirmation",
                "Evaluate transplant candidacy",
            ],
        }
    if stage in ("F1", "F2") and risk_label == "HIGH":
        return {
            "severity": "MODERATE",
            "recommendation": (
                f"MODERATE: Early fibrosis ({stage}, {stage_conf:.0%}) but temporal trajectory "
                f"shows HIGH risk ({risk_prob:.0%}). Close surveillance needed."
            ),
            "actions": [
                "Hepatology follow-up in 2 weeks",
                "Lifestyle modification counseling",
                "Repeat labs in 14 days",
                "Consider liver biopsy",
            ],
        }
    if stage in ("F1", "F2"):
        return {
            "severity": "LOW-MODERATE",
            "recommendation": (
                f"LOW-MODERATE: {stage} fibrosis ({stage_conf:.0%}), "
                f"{risk_label} temporal risk ({risk_prob:.0%}). Continue monitoring."
            ),
            "actions": [
                "Routine follow-up in 3-6 months",
                "Annual FibroScan",
                "Lifestyle counseling",
            ],
        }
    if stage == "F0":
        return {
            "severity": "LOW",
            "recommendation": (
                f"LOW RISK: No significant fibrosis ({stage_conf:.0%}). "
                f"Temporal risk: {risk_label} ({risk_prob:.0%}). Routine surveillance."
            ),
            "actions": [
                "Annual liver function panel",
                "Lifestyle counseling",
                "Re-image in 12-24 months if risk factors persist",
            ],
        }
    return {
        "severity": "INDETERMINATE",
        "recommendation": f"Stage: {stage}, Risk: {risk_label}. Hepatology consultation recommended.",
        "actions": ["Hepatology consultation", "Complete workup"],
    }

# =============================================================================
# RUN BUTTON
# =============================================================================
run_btn = st.button("Run HepSense CDSS", type="primary", use_container_width=True)

if run_btn:
    if uploaded_image is None and df_ehr is None:
        st.warning("Please upload at least one input (ultrasound image or EHR CSV).")
        st.stop()

    # -- Vision Expert --
    vision_stage = "N/A"
    vision_conf = 0.0
    vision_probs = {}
    gradcam_img = None

    if uploaded_image is not None and vision_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.getvalue())
            tmp_path = tmp.name

        with st.spinner("Running Vision Expert (DANN EfficientNet-B0)..."):
            vis_result = predict_ultrasound(tmp_path, vision_model)
            vision_stage = vis_result["predicted_stage"]
            vision_conf = vis_result["confidence"]
            vision_probs = vis_result["probabilities"]
            gradcam_img = vis_result["gradcam_overlay"]
        os.unlink(tmp_path)
    elif uploaded_image is not None and vision_model is None:
        st.warning("Vision model not found. Run HepSense_Vision.py first.")

    # -- t-MELD Expert --
    tmeld_prob = 0.0
    tmeld_label = "N/A"
    tmeld_ran = False

    if df_ehr is not None and tmeld_model is not None:
        with st.spinner("Running t-MELD Temporal Risk Engine (tsfresh + XGBoost)..."):
            try:
                tmeld_prob, tmeld_label = run_tmeld_pipeline(df_ehr, age_val, gender_val)
                tmeld_ran = True
            except Exception as e:
                st.error(f"t-MELD pipeline error: {e}")
                import traceback
                st.expander("Traceback").text(traceback.format_exc())
    elif df_ehr is not None and tmeld_model is None:
        st.warning("t-MELD model not found. Run train.py first.")

    # -- Integration --
    if vision_stage != "N/A" or tmeld_ran:
        rec = integration_engine(
            vision_stage if vision_stage != "N/A" else "F0",
            vision_conf,
            tmeld_label if tmeld_ran else "LOW",
            tmeld_prob,
        )

        # Severity banner
        css_class = rec["severity"].replace(" ", "-")
        st.markdown(f"""
        <div class="severity-card severity-{css_class}">
            <div class="severity-label">HepSense Combined Assessment</div>
            <div class="severity-level">{rec['severity']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Expert cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="expert-card">
                <h3>Vision Expert (DANN)</h3>
                <div class="value">{vision_stage}</div>
                <div class="sub">{'Confidence: ' + f'{vision_conf:.1%}' if vision_stage != 'N/A' else 'No image uploaded'}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="expert-card">
                <h3>t-MELD Temporal Risk</h3>
                <div class="value">{tmeld_label}</div>
                <div class="sub">{'14-Day Risk: ' + f'{tmeld_prob:.1%}' if tmeld_ran else 'No EHR uploaded'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recommendation
        st.markdown("### Clinical Recommendation")
        st.markdown(f'<div class="rec-box">{rec["recommendation"]}</div>', unsafe_allow_html=True)

        st.markdown("### Recommended Actions")
        for a in rec["actions"]:
            st.markdown(f'<div class="action-item">-> {a}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Grad-CAM + Probabilities
        col_cam, col_prob = st.columns(2)
        with col_cam:
            st.markdown("### Grad-CAM Explainability")
            if gradcam_img is not None:
                st.image(gradcam_img, caption=f"Grad-CAM: {vision_stage} ({vision_conf:.1%})",
                         use_container_width=True)
            else:
                st.info("Upload an ultrasound image to see Grad-CAM heatmap.")

        with col_prob:
            st.markdown("### Stage Probabilities")
            if vision_probs:
                stages = list(vision_probs.keys())
                values = list(vision_probs.values())
                colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(stages, values, color=colors, edgecolor="white", linewidth=1.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", fontsize=12)
                ax.invert_yaxis()
                for bar, val in zip(bars, values):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                            f"{val:.1%}", va="center", fontsize=11, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif tmeld_ran:
                # Show t-MELD risk gauge instead
                fig, ax = plt.subplots(figsize=(6, 4))
                color = "#ef4444" if tmeld_prob >= 0.5 else ("#eab308" if tmeld_prob >= 0.2 else "#22c55e")
                ax.barh(["14-Day\nDecompensation"], [tmeld_prob], color=color, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Calibrated Probability")
                ax.text(tmeld_prob + 0.03, 0, f"{tmeld_prob:.1%}", va="center", fontsize=14, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
