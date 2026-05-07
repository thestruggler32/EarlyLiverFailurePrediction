# HepSense (DeepHepaStrat)

**An AI-driven Risk Stratification System for Early Liver Failure Prediction**

<div align="center">
  <em>Shifting the clinical paradigm from static "disease staging" to dynamic, longitudinal future prediction.</em>
</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [The Clinical Problem](#-the-clinical-problem)
- [Methodology](#-methodology)
  - [1. Vision Pipeline (`HepSense_Vision.py`)](#1-vision-pipeline-hepsense_visionpy)
  - [2. Clinical Temporal Pipeline (MIMIC-IV)](#2-clinical-temporal-pipeline-mimic-iv)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Literature & Background](#-literature--background)
- [Development Timeline](#-development-timeline)
- [Expected Outcomes](#-expected-outcomes)

---

## 🧬 Project Overview

Liver cirrhosis is the 11th leading cause of death globally (over 1.3 million deaths annually). 
**HepSense (DeepHepaStrat)** is designed as a multi-modal framework that integrates both computer vision (for imaging analysis) and time-series/NLP data (for clinical health records) to provide a holistic, dynamic risk assessment of liver cirrhosis. It acts as a temporal AI "early warning radar" to reduce preventable ICU mortality by predicting acute decompensation events before they occur.

## ⚠️ The Clinical Problem

Current clinicians rely on outdated, static scoring equations (e.g., MELD-Na, FIB-4). 
* **Snapshot Bias:** Static scores analyze a single day's blood test, creating a dangerous bias that misses declining trajectories.
* **Blind Spots:** Doctors are often blind to impending, fatal crises like variceal hemorrhage or encephalopathy.
* **Reactive vs Proactive:** Most existing models predict mortality *after* a patient enters critical condition in the ICU. HepSense bridges this gap by forecasting the actual future onset of an emergency **7 to 14 days before it happens**, allowing for preventative intervention.

---

## 🔬 Methodology

### 1. Vision Pipeline (`HepSense_Vision.py`)
A deep learning computer vision model built to classify the progression of liver cirrhosis stages (F0 to F4) from medical imaging. 
* **Architecture:** Utilizes a pre-trained `MobileNetV2` as the core feature extractor with custom dense classification layers.
* **Techniques:** Implements data augmentation to handle class imbalances, dropout layers to prevent overfitting, and sparse categorical crossentropy for robust multiclass classification.

### 2. Clinical Temporal Pipeline (MIMIC-IV)
A quantitative, retrospective data-mining structure utilizing multi-modal clinical histories to capture the velocity of a patient's decline over a 14-day window.
* **Data Collection:** Executes authorized, secure extraction of de-identified critical care records directly from MIT's PhysioNet repository (`mimic-hosp`, `mimic-icu`, `mimic-ed`, `mimic-note`).
* **Data Analysis & NLP:** 
  - Missing-value imputation via "Last Observation Carried Forward" (LOCF).
  - NLP (HuggingFace transformers & spaCy) to extract unstructured symptom features (e.g., "vomiting blood") from clinical admission notes.
* **Predictive Modeling:** Time-Series Gradient Boosted Trees (XGBoost) / LSTMs for forecasting, cross-validated against held-out testing cohorts.
* **Explainability (XAI):** Employs SHAP (SHapley Additive exPlanations) values to visualize feature importance, ensuring absolute medical transparency and physician trust.

---

## 🚀 Getting Started

### Prerequisites

This project uses modern Python tools and depends heavily on deep learning libraries. It is managed using `uv` for lightning-fast dependency resolution.

* Python >= 3.12
* `uv` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd EarlyLiverFailurePrediction
   ```

2. **Sync the environment:**
   Use `uv` to install dependencies as defined in the `pyproject.toml` / `uv.lock`.
   ```bash
   uv sync
   ```

3. **Dependencies:**
   The project requires the following core libraries (see `requirements.txt`):
   * `tensorflow` (>= 2.13.0)
   * `numpy`, `pandas`, `scikit-learn`, `scipy`
   * `xgboost`, `catboost`, `lightgbm`
   * `matplotlib`, `seaborn`
   * `flask`

---

## 💻 Usage

### Running the Vision Model Training

The current working implementation includes the Vision Pipeline. To train the MobileNetV2-based model on your image dataset:

1. Ensure your dataset is structured correctly in the `Dataset/` directory, with subfolders representing the cirrhosis stages (e.g., `F0`, `F1`, `F2`, `F3`, `F4`).
2. Run the vision script:
   ```bash
   python HepSense_Vision.py
   ```
3. The script will output the training and validation progress and save the finalized model locally as `hepsense_vision_v1.keras`.

---

## 📚 Literature & Background

Our theoretical framework, **Explainable Temporal Machine Learning Theory**, posits that physiological collapse in chronic end-stage liver disease is not sudden; it is preceded by subtle, long-term degradation across multiple routine biomarkers.

| Author | Title & Inference | Research Gap Addressed |
| :--- | :--- | :--- |
| **Kumar, A., et al.** | *A Deep Learning Framework for Forecasting Hepatic Decompensation...* LSTM on 14-day trailing EHR identified bleeding early vs single-day MELD score. | We use an accessible dataset (MIMIC-IV) for broader replication rather than a private hospital database. |
| **Zhang, L., et al.** | *Predicting Mortality in Patients with Liver Cirrhosis Using Machine Learning...* XGBoost on MIMIC-IV predicted mortality with AUC 0.86 using bedside labs. | We focus on dynamic time-of-onset prediction rather than 24-hour snapshots. |
| **Martinez, R., et al.** | *Explainable Artificial Intelligence for Clinical Risk Stratification...* Black-box ML models are rejected unless SHAP/LIME explains features to doctors. | We integrate dynamic time-series logic seamlessly with XAI. |
| **Chen, S., et al.** | *Multimodal Fusion of Clinical and Imaging Data for Sarcopenia Detection...* Automating SMI calculations from CT scans dramatically improves mortality forecasts. | We fuse imaging capabilities with sequential blood labs and NLP symptom tracking. |

### Trends in Literature
- **Static to Dynamic:** Rapid movement away from fixed clinical equations (MELD) toward algorithms that process sequential, time-series data.
- **Interpretable AI (XAI):** Strict requirement for local explanations (like SHAP) in medical journals to increase physician adoption.
- **NLP Integration:** Critical predictive signals exist in the unstructured text of doctors' notes, prompting the fusion of NLP with standard numeric lab data.

---

## 📅 Development Timeline

1. **Initiation (Weeks 1-2):** Define objectives, secure CITI training, and gain MIMIC-IV access. Configure virtual environments.
2. **Planning (Weeks 3-4):** Map SQL extraction for MIMIC-IV tables and finalize ML architecture (XGBoost vs. LSTM algorithms).
3. **Execution (Weeks 5-10):** Execute codebase for missing-value imputation, program NLP modules for unstructured notes, and train baseline models. Integrate SHAP into the Flask web dashboard.
4. **Closure (Weeks 11-13):** Benchmark against MELD score, finalize the frontend dashboard, and draft the final research paper for presentation.

---

## 🎯 Expected Outcomes

* **Target Performance:** Expected AUROC of >0.85, establishing proven, quantitative superiority over the static MELD-Na clinical score.
* **Publications:** Targeting premier bioinformatics conferences (e.g., IEEE EMBS or BHI) and high-impact hepatology journals (e.g., IEEE JBHI).
* **Patents:** Formal patent filing for the explicit architectural fusion of the SHAP dashboard and LSTM temporal forecasting.
* **Clinical Impact:** Empowers clinicians to schedule preemptive, prophylactic interventions safely (e.g., executing an endoscopy before a fatal hemorrhage occurs), optimizing hospital triage resources.
