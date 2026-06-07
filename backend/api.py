import sys
import os
import io
import base64
import torch
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_trained_model, run_vision_expert, run_clinical_expert, run_temporal_expert, integration_engine
from HepSense_Vision import CLASS_NAMES, DEVICE, MODEL_SAVE_PATH

app = FastAPI(title="HepSense CDSS Core Engine", description="API for HepSense Liver Risk Assessment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to prevent re-instantiation overhead
VISION_MODEL = None
TEMPORAL_MODEL = None
CLINICAL_MODEL = None
SELECTED_FEATURES = []

@app.on_event("startup")
def load_engines():
    global VISION_MODEL, TEMPORAL_MODEL, CLINICAL_MODEL, SELECTED_FEATURES
    try:
        # Load XGBoost Ensembles
        TEMPORAL_MODEL = joblib.load("../hepsense_temporal_xgboost_v1.joblib")
        CLINICAL_MODEL = joblib.load("../hepsense_clinical_xgboost_v1.joblib")
        
        # Load expected features
        SELECTED_FEATURES = joblib.load("../selected_features.pkl")
        
        # Load DenseNet121 Architecture & Weights
        VISION_MODEL = load_trained_model("../hepsense_vision_dann_v2.pth")
        
        print(f"[SUCCESS] Multimodal engines loaded onto {DEVICE} successfully.")
    except Exception as e:
        print(f"[CRITICAL] Component initialization failed: {str(e)}")


def extract_temporal_trajectories(df: pd.DataFrame, age: int, gender: str) -> dict:
    """
    Transforms raw longitudinal lab sequences into the exact 34 feature vector
    expected by the trained temporal XGBoost classifier.
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    
    if 'lab_test_name' not in df.columns:
        # If it's a wide format, melt it
        id_vars = ['charttime'] if 'charttime' in df.columns else [cols_lower.get('charttime', 'charttime')]
        df = df.melt(id_vars=id_vars, var_name='lab_test_name', value_name='valuenum')
        
    df['patient_id'] = 1  # Force single-patient ID context
    charttime_col = cols_lower.get('charttime', 'charttime')
    if charttime_col in df.columns:
        df = df.rename(columns={charttime_col: 'charttime'})
    
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df = df.dropna(subset=['charttime', 'valuenum'])
    
    target_tests = [
        "Bilirubin, Total", "INR(PT)", "Creatinine",
        "Platelet Count", "Alanine Aminotransferase (ALT)",
        "Asparate Aminotransferase (AST)",
    ]
    df = df[df['lab_test_name'].isin(target_tests)]
    
    df = df.sort_values('charttime')
    df_pivot = df.pivot_table(index=['patient_id', 'charttime'], columns='lab_test_name', values='valuenum').reset_index()
    
    # Handle missing values for tsfresh
    for test in target_tests:
        if test not in df_pivot.columns:
            df_pivot[test] = 1.0 # Standard default
    df_pivot = df_pivot.ffill().fillna(1.0)
    
    last_obs = df_pivot.groupby('patient_id').last()
    
    def row_meld(row):
        cr = max(row.get('Creatinine', 1.0) if not pd.isna(row.get('Creatinine', 1.0)) else 1.0, 1.0)
        tb = max(row.get('Bilirubin, Total', 1.0) if not pd.isna(row.get('Bilirubin, Total', 1.0)) else 1.0, 1.0)
        inr = max(row.get('INR(PT)', 1.0) if not pd.isna(row.get('INR(PT)', 1.0)) else 1.0, 1.0)
        score = 3.78 * np.log(tb) + 11.2 * np.log(inr) + 9.57 * np.log(cr) + 6.43
        return np.clip(score, 6, 40)
        
    def row_fib4(row):
        a = age
        ast = row.get('Asparate Aminotransferase (AST)', 40)
        alt = row.get('Alanine Aminotransferase (ALT)', 40)
        plt = row.get('Platelet Count', 150)
        if pd.isna(ast) or pd.isna(alt) or pd.isna(plt) or plt == 0:
            return 1.0
        return (a * ast) / (plt * np.sqrt(alt))
        
    meld = row_meld(last_obs.iloc[0]) if len(last_obs) > 0 else 6.43
    fib4 = row_fib4(last_obs.iloc[0]) if len(last_obs) > 0 else 1.0
    
    # tsfresh extraction (optimized via n_jobs=0 for single patient)
    extracted_df = extract_features(
        df_pivot, 
        column_id='patient_id', 
        column_sort='charttime',
        disable_progressbar=True,
        n_jobs=0,
        default_fc_parameters=ComprehensiveFCParameters()
    )
    
    # Add static features
    extracted_df['MELD_Score'] = meld
    extracted_df['FIB4_Score'] = fib4
    extracted_df['age'] = age
    extracted_df['sex'] = 1 if gender.upper() == 'M' else 0
    extracted_df['NLP_Encephalopathy_Flag'] = 0
    extracted_df['NLP_Variceal_Bleeding_Flag'] = 0
    
    # Reindex to exact 34 features
    final_features = extracted_df.reindex(columns=SELECTED_FEATURES, fill_value=0.0)
    
    return final_features.iloc[0].to_dict()

def encode_image(img_array):
    if img_array is None:
        return None
    from PIL import Image
    img = Image.fromarray(np.uint8(img_array * 255) if img_array.dtype == np.float32 else img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/api/analyze")
async def analyze(
    age: int = Form(55),
    gender: str = Form("M"),
    image: UploadFile = File(None),
    csv_file: UploadFile = File(None)
):
    vision_result = {"stage": "N/A", "confidence": 0.0, "probabilities": {}, "gradcam_overlay": None}
    clinical_result = {"risk_probability": 0.0, "risk_label": "UNAVAILABLE", "error": None}
    temporal_result = {"trend_risk": 0.0, "trend_label": "UNAVAILABLE", "failing_point": "Unknown", "error": None}
    
    if image:
        content = await image.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        vision_result = run_vision_expert(tmp_path, VISION_MODEL)
        os.unlink(tmp_path)
        
        if "gradcam_overlay" in vision_result and vision_result["gradcam_overlay"] is not None:
            vision_result["gradcam_overlay"] = encode_image(vision_result["gradcam_overlay"])

    trends_data = [] # Keeping trends_data empty for now to avoid frontend crash if missing
    if csv_file:
        content = await csv_file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
            
            temporal_features = extract_temporal_trajectories(df, age, gender)
            
            clinical_result = run_clinical_expert(temporal_features, CLINICAL_MODEL)
            temporal_result = run_temporal_expert(temporal_features, TEMPORAL_MODEL)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV processing error: {str(e)}")
            
    recommendation = integration_engine(vision_result, clinical_result, temporal_result)
    
    return {
        "vision": vision_result,
        "clinical": clinical_result,
        "temporal": temporal_result,
        "recommendation": recommendation,
        "trends": trends_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
