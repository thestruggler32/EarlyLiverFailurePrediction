"""
HepSense -- Clinical Decision Support Dashboard
================================================
Clinician-friendly interface for liver disease risk assessment.
Combines ultrasound imaging analysis with longitudinal lab trends
to stratify patients by decompensation risk.

Usage:  streamlit run streamlit_app.py
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
    page_title="HepSense - Liver Risk Assessment System",
    page_icon="\u2695",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- HepSense modules --
from HepSense_Vision import (
    HepSenseDANN, load_trained_model, predict_ultrasound,
    CLASS_NAMES, DEVICE, MODEL_SAVE_PATH,
)

# =============================================================================
# CSS  -- Clinical-grade styling
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, sans-serif; }

    /* -- Top header bar -- */
    .clinical-header {
        background: #0a2942; border-bottom: 4px solid #1a7f6e;
        padding: 14px 28px; margin-bottom: 20px; color: white;
        display: flex; justify-content: space-between; align-items: center;
    }
    .clinical-header .title { font-size: 20px; font-weight: 600; }
    .clinical-header .sub  { font-size: 12px; opacity: 0.7; }
    .clinical-header .badge {
        background: #1a7f6e; padding: 4px 14px; border-radius: 4px;
        font-size: 11px; font-weight: 500; letter-spacing: 0.3px;
    }

    /* -- Sidebar patient info -- */
    .patient-card {
        background: #f0f4f8; border: 1px solid #dce3ed; border-radius: 6px;
        padding: 14px 16px; margin-bottom: 16px;
    }
    .patient-card .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .patient-card .value { font-size: 14px; font-weight: 600; color: #0f172a; }

    /* -- Severity alert banner -- */
    .alert-banner {
        padding: 16px 24px; border-radius: 6px; margin: 12px 0;
        border-left: 6px solid; font-size: 15px; line-height: 1.5;
    }
    .alert-CRITICAL       { background: #fef2f2; border-color: #dc2626; color: #7f1d1d; }
    .alert-HIGH           { background: #fff7ed; border-color: #f97316; color: #7c2d12; }
    .alert-MODERATE       { background: #fefce8; border-color: #eab308; color: #713f12; }
    .alert-LOW-MODERATE   { background: #f0fdf4; border-color: #84cc16; color: #3f6212; }
    .alert-LOW            { background: #f0fdfa; border-color: #14b8a6; color: #134e4a; }
    .alert-INDETERMINATE  { background: #f8fafc; border-color: #94a3b8; color: #334155; }

    .alert-banner .alert-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .alert-banner .alert-level { font-size: 26px; font-weight: 700; margin: 2px 0; }

    /* -- Metric cards -- */
    .metric-card {
        background: white; border: 1px solid #e5e9f0; border-radius: 6px;
        padding: 16px; text-align: center;
    }
    .metric-card h3    { margin: 0 0 2px 0; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #0f172a; }
    .metric-card .sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }

    /* -- Recommendation box -- */
    .rec-box {
        background: #f0f4f8; border: 1px solid #dce3ed; border-radius: 6px;
        padding: 14px 18px; font-size: 14px; line-height: 1.6; color: #1e293b; margin: 10px 0;
    }

    /* -- Action items -- */
    .action-item {
        background: white; border: 1px solid #e5e9f0; border-radius: 4px;
        padding: 8px 14px; margin: 4px 0; font-size: 13px; color: #334155;
    }

    /* -- Input sections -- */
    .input-section {
        background: white; border: 1px solid #e5e9f0; border-radius: 6px;
        padding: 18px; min-height: 260px;
    }
    .input-section h3 { margin: 0 0 4px 0; font-size: 14px; color: #0f172a; }
    .input-section .hint { font-size: 12px; color: #94a3b8; margin-bottom: 12px; }

    /* -- Section divider -- */
    .section-label {
        font-size: 13px; font-weight: 600; color: #0f172a;
        border-bottom: 1px solid #e5e9f0; padding-bottom: 6px; margin: 18px 0 10px 0;
    }

    .st-emotion-cache-1y4p8pa { padding: 1.5rem 1rem; }
    .st-emotion-cache-16txtl3 { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="clinical-header">
    <div>
        <div class="title">HepSense &mdash; Liver Risk Assessment</div>
        <div class="sub">Clinical Decision Support &bull; Risk Stratification for Cirrhotic Patients</div>
    </div>
    <div class="badge">For Professional Use</div>
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
        ts_cols = [c for c in selected_cols if '__' in c]
        kind_to_fc = from_columns(ts_cols)
        return model, selected_cols, le, kind_to_fc
    except Exception as e:
        st.error(f"t-MELD artifacts missing: {e}")
        return None, None, None, None

vision_model = load_vision_model()
tmeld_model, selected_cols, le, kind_to_fc = load_tmeld_artifacts()

# =============================================================================
# SIDEBAR -- Patient Information
# =============================================================================
with st.sidebar:
    st.markdown("### Patient Information")
    st.markdown(f"**Patient ID**: P-{np.random.randint(10000,99999)}")
    c1, c2 = st.columns(2)
    with c1:
        age_val = st.number_input("Age", 18, 120, 55)
    with c2:
        if le is not None:
            gender_val = st.selectbox("Sex", le.classes_)
        else:
            gender_val = st.selectbox("Sex", ["M", "F"])

    st.divider()

    st.markdown("### Input Data")

    st.markdown("**Ultrasound Image**")
    uploaded_image = st.file_uploader(
        "Upload liver ultrasound", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed", key="img_upload",
    )
    if uploaded_image:
        preview = Image.open(uploaded_image)
        st.image(preview, use_container_width=True)

    st.markdown("**Lab Results (CSV)**")
    ehr_file = st.file_uploader(
        "Upload EHR CSV", type=["csv"],
        label_visibility="collapsed",
        help="CSV with columns: charttime, lab_test_name, valuenum",
        key="ehr_upload",
    )
    if ehr_file:
        try:
            df_ehr = pd.read_csv(ehr_file)
            st.caption(f"{len(df_ehr)} rows loaded")
        except Exception as e:
            st.error(f"CSV error: {e}")

    # Quick paste fallback
    if not ehr_file:
        pasted = st.text_area(
            "Or paste CSV data", height=80, label_visibility="collapsed",
            placeholder='charttime,Bilirubin Total,INR(PT),Creatinine\n2023-10-01,1.2,1.1,0.9',
        )
        if pasted:
            try:
                sep = ',' if ',' in pasted.split('\n')[0] else '\t'
                df_ehr = pd.read_csv(io.StringIO(pasted), sep=sep)
                st.caption(f"{len(df_ehr)} rows parsed")
            except Exception:
                st.error("Parse failed.")

    st.caption("This tool provides risk estimates to aid clinical decision-making. All results should be reviewed by a qualified hepatologist.")

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
    Process lab data and compute decompensation risk score.
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

    return prob, label, X_final

# =============================================================================
# Integration Engine
# =============================================================================
def integration_engine(stage, stage_conf, risk_label, risk_prob):
    """Combine fibrosis stage and decompensation risk into unified recommendation."""

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
run_btn = st.button("Run Risk Assessment", type="primary", use_container_width=True)

# Run indicator
placeholder = st.empty()

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

        with st.spinner("Analyzing ultrasound image..."):
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
        with st.spinner("Analyzing lab trends and calculating risk..."):
            try:
                tmeld_prob, tmeld_label, X_final = run_tmeld_pipeline(df_ehr, age_val, gender_val)
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

        # -- CLINICAL ALERT BANNER --
        css_class = rec["severity"].replace(" ", "-")
        st.markdown(f"""
        <div class="alert-banner alert-{css_class}">
            <div class="alert-title">HepSense Risk Classification</div>
            <div class="alert-level">{rec['severity']}</div>
            <div style="font-size:13px;margin-top:4px;">{rec['recommendation']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">CLINICAL MEASURES</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Fibrosis Stage</h3>
                <div class="value">{vision_stage}</div>
                <div class="sub">{'Model confidence: ' + f'{vision_conf:.1%}' if vision_stage != 'N/A' else 'No image uploaded'}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Decompensation Risk</h3>
                <div class="value">{tmeld_label}</div>
                <div class="sub">{'14-Day probability: ' + f'{tmeld_prob:.1%}' if tmeld_ran else 'No lab data uploaded'}</div>
            </div>
            """, unsafe_allow_html=True)

        # -- RECOMMENDED ACTIONS --
        st.markdown('<div class="section-label">MANAGEMENT PLAN</div>', unsafe_allow_html=True)
        for a in rec["actions"]:
            st.markdown(f'<div class="action-item">\u2022 {a}</div>', unsafe_allow_html=True)

        # -- DIAGNOSTIC RESULTS --
        st.markdown('<div class="section-label">DIAGNOSTIC FINDINGS</div>', unsafe_allow_html=True)
        col_cam, col_prob = st.columns(2)
        with col_cam:
            st.markdown("**Ultrasound Analysis**")
            if gradcam_img is not None:
                st.image(gradcam_img, caption=f"Fibrosis stage: {vision_stage}  |  Confidence: {vision_conf:.1%}",
                         use_container_width=True)
            else:
                st.info("Upload an ultrasound image to view the analysis.")

        with col_prob:
            st.markdown("**Stage Likelihood**")
            if vision_probs:
                stages = list(vision_probs.keys())
                values = list(vision_probs.values())
                colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                bars = ax.barh(stages, values, color=colors, edgecolor="white", linewidth=1.2)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.invert_yaxis()
                for bar, val in zip(bars, values):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                            f"{val:.1%}", va="center", fontsize=10, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif tmeld_ran:
                fig, ax = plt.subplots(figsize=(5.5, 2.5))
                color = "#dc2626" if tmeld_prob >= 0.5 else ("#eab308" if tmeld_prob >= 0.2 else "#14b8a6")
                ax.barh(["14-Day\nDecompensation\nRisk"], [tmeld_prob], color=color, height=0.35)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Risk Score")
                ax.text(tmeld_prob + 0.03, 0, f"{tmeld_prob:.1%}", va="center", fontsize=12, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # -- LAB TRENDS + BIOMARKER ANALYSIS --
        if tmeld_ran:
            st.markdown('<div class="section-label">LABORATORY TRENDS</div>', unsafe_allow_html=True)
            col_trends, col_shap = st.columns(2)
            
            with col_trends:
                st.markdown("**Serial Lab Values (Normalized)**")
                labs_avail = False
                if 'lab_test_name' in df_ehr.columns:
                    plot_df = df_ehr.copy()
                    plot_df['charttime'] = pd.to_datetime(plot_df['charttime'])
                    plot_df = plot_df.pivot_table(index='charttime', columns='lab_test_name', values='valuenum')
                    labs_avail = True
                elif df_ehr is not None:
                    plot_df = df_ehr.copy()
                    plot_df['charttime'] = pd.to_datetime(plot_df.get('charttime', plot_df.index))
                    plot_df.set_index('charttime', inplace=True)
                    labs_avail = True
                
                cols_to_plot = [c for c in TARGET_TESTS if c in plot_df.columns] if labs_avail else []
                if cols_to_plot:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    for c in cols_to_plot:
                        series = plot_df[c].dropna()
                        if not series.empty:
                            ax.plot(series.index, series / series.max(), marker='o', label=c, linewidth=1.5)
                    ax.set_title("Lab Trends (Normalized)")
                    ax.legend(fontsize=9)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Serial lab data not available for plotting.")

            with col_shap:
                st.markdown("**Key Risk Drivers**")
                try:
                    import shap
                    
                    base_model = tmeld_model
                    if hasattr(tmeld_model, 'estimator'):
                        base_model = tmeld_model.estimator
                        if hasattr(base_model, 'estimator'):
                            base_model = base_model.estimator
                    elif hasattr(tmeld_model, 'calibrated_classifiers_'):
                        base_model = tmeld_model.calibrated_classifiers_[0].estimator
                        if hasattr(base_model, 'estimator'):
                            base_model = base_model.estimator

                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_final)
                    
                    fig, ax = plt.subplots(figsize=(7, 4))
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                        base_val = explainer.expected_value[1]
                    else:
                        sv = shap_values[0]
                        base_val = explainer.expected_value
                    
                    shap.plots._waterfall.waterfall_legacy(base_val, sv, feature_names=X_final.columns.tolist(), max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Risk factor analysis unavailable: {e}")

