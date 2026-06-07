import requests
import pandas as pd
import numpy as np
import io
import shutil

# 1. Create Synthetic Crashing Labs (CSV)
data = {
    'patient_id': [1, 1, 1, 1, 1, 1],
    'charttime': pd.date_range(start='2026-06-01', periods=6, freq='D'),
    'lab_test_name': ['Bilirubin, Total']*3 + ['Platelet Count']*3,
    'valuenum': [1.2, 5.8, 15.4,  # Bilirubin skyrocketing
                 150, 80, 20]      # Platelets plummeting
}
df = pd.DataFrame(data)
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode('utf-8')

# 2. Get an F0 Ultrasound Image
f0_image_path = r"d:\EarlyLiverFailurePredictionN\EarlyLiverFailurePrediction\Ultrasonic_dataset\Dataset\Dataset\F0\a1000.jpg"
# Try to find one if the exact name differs
import os
import glob
f0_files = glob.glob(r"d:\EarlyLiverFailurePredictionN\EarlyLiverFailurePrediction\Ultrasonic_dataset\Dataset\Dataset\F0\*.jpg")
if f0_files:
    f0_image_path = f0_files[0]

# 3. Hit the API
url = "http://127.0.0.1:8000/api/analyze"
files = {
    'csv_file': ('crashing_labs.csv', csv_bytes, 'text/csv'),
    'image': ('f0_ultrasound.jpg', open(f0_image_path, 'rb'), 'image/jpeg')
}
data = {
    'age': 45,
    'gender': 'M'
}

print("Firing API Request...")
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    res = response.json()
    print("\n--- API RESPONSE ---")
    print(f"Vision Stage: {res['vision']['stage']} (Conf: {res['vision']['confidence']:.2f})")
    print(f"Clinical Risk: {res['clinical']['risk_label']} (Prob: {res['clinical']['risk_probability']:.2f})")
    print(f"Temporal Trend: {res['temporal']['trend_label']} (Risk: {res['temporal']['trend_risk']:.2f})")
    print("\n=== INTEGRATION ENGINE RECOMMENDATION ===")
    print(f"Severity Level: {res['recommendation']['severity_level']}")
    print(f"Summary: {res['recommendation']['recommendation']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
