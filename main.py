import streamlit as st
import pickle
import pandas as pd
import base64
from pathlib import Path

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    try:
        pipe = pickle.load(open(r"C:\MY\WORK\PROJECTS\Diabetes\model.pkl", 'rb'))
        return pipe, list(pipe.feature_names_in_)
    except FileNotFoundError:
        # Fallback for deployment - use a dummy model if file not found
        st.warning("⚠️ Model file not found. Using fallback mode.")
        return None, ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

pipe, feature_order = load_model()

# ------------------- CUSTOM STYLING -------------------
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f33 50%, #0d1326 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Animated gradient background */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 50%, rgba(30, 136, 229, 0.05) 0%, transparent 70%),
                    radial-gradient(ellipse at 70% 80%, rgba(67, 160, 71, 0.04) 0%, transparent 60%);
        animation: rotateBg 30s linear infinite;
        z-index: 0;
        pointer-events: none;
    }
    
    @keyframes rotateBg {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Main container */
    .main-container {
        position: relative;
        z-index: 1;
        max-width: 820px;
        margin: 0 auto;
        padding: 1.5rem 1rem;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(18, 25, 45, 0.75);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 32px;
        padding: 2.5rem 2.8rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8),
                    0 0 0 1px rgba(255, 255, 255, 0.03) inset;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(79, 195, 247, 0.03) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    
    /* Header */
    .header-wrapper {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.5rem;
    }
    
    .header-icon {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #4fc3f7, #1e88e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 20px rgba(79, 195, 247, 0.2));
    }
    
    .main-title {
        color: #f0f4fc;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e8f0fe, #b0d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .subtitle {
        color: #889bb8;
        font-size: 1rem;
        padding-left: 4px;
        border-left: 3px solid #1e88e5;
        padding-left: 16px;
        background: rgba(30, 136, 229, 0.06);
        border-radius: 0 20px 20px 0;
        padding: 10px 18px;
        margin: 0.5rem 0 1.8rem 0;
        display: inline-block;
        font-weight: 400;
    }
    
    .subtitle i {
        margin-right: 10px;
        color: #4fc3f7;
        -webkit-text-fill-color: #4fc3f7;
    }
    
    /* Input styling */
    .stNumberInput > div {
        position: relative;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(12, 20, 35, 0.7) !important;
        border: 1px solid #2a3d5a !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #e8f0fe !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(4px) !important;
        height: 48px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4fc3f7 !important;
        background: rgba(20, 35, 60, 0.85) !important;
        box-shadow: 0 0 0 3px rgba(79, 195, 247, 0.12), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stNumberInput > div > div > input::placeholder {
        color: #5a7088 !important;
        font-weight: 300 !important;
    }
    
    /* Labels */
    .stNumberInput > label {
        color: #cbdae9 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 4px !important;
    }
    
    /* Grid layout for inputs */
    .input-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 1.2rem 1.8rem !important;
        margin: 1.5rem 0 2rem 0 !important;
    }
    
    .input-grid > div {
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Button styling */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        margin: 0.5rem 0 1rem 0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(145deg, #1a73e8, #0d47a1) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 14px 44px !important;
        border-radius: 60px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(26, 115, 232, 0.25) !important;
        letter-spacing: 0.5px !important;
        gap: 12px !important;
        width: auto !important;
        min-width: 220px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
        transition: left 0.6s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(26, 115, 232, 0.4) !important;
        background: linear-gradient(145deg, #2a7de1, #0f4a9e) !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:active {
        transform: scale(0.97) !important;
    }
    
    /* Result styling */
    .result-box {
        margin-top: 1.5rem !important;
        border-radius: 20px !important;
        padding: 1.2rem 1.8rem !important;
        background: rgba(10, 20, 35, 0.6) !important;
        border-left: 6px solid #4fc3f7 !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(79, 195, 247, 0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    .result-box .stAlert {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .result-box .stAlert > div {
        background: transparent !important;
        padding: 0 !important;
        font-size: 1.05rem !important;
    }
    
    .result-box .stAlert .st-emotion-cache-1y4p8pa {
        background: transparent !important;
    }
    
    /* Custom result display */
    .result-high-risk {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.12), rgba(255, 82, 82, 0.04)) !important;
        border-left: 6px solid #ff5252 !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.8rem !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, rgba(105, 219, 124, 0.12), rgba(105, 219, 124, 0.04)) !important;
        border-left: 6px solid #69db7c !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.8rem !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .result-status {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
    }
    
    .result-prob {
        font-size: 1rem !important;
        color: #98b1cc !important;
        margin-top: 6px !important;
    }
    
    /* Help text styling */
    .stNumberInput .st-emotion-cache-1n76uvr {
        color: #5a7088 !important;
        font-size: 0.75rem !important;
        margin-top: 2px !important;
        font-weight: 300 !important;
    }
    
    /* Responsive */
    @media (max-width: 640px) {
        .glass-card {
            padding: 1.8rem 1.2rem !important;
            border-radius: 24px !important;
        }
        
        .input-grid {
            grid-template-columns: 1fr !important;
            gap: 0.8rem !important;
        }
        
        .main-title {
            font-size: 1.8rem !important;
        }
        
        .stButton > button {
            width: 100% !important;
            min-width: unset !important;
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0e1a;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a4b6e;
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- MAIN UI -------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="glass-card">
        <div class="header-wrapper">
            <span class="header-icon">🫀</span>
            <h1 class="main-title">Diabetes Risk Predictor</h1>
        </div>
        <div class="subtitle">
            <span>🧬</span> Enter patient metrics · AI-powered risk assessment
        </div>
""", unsafe_allow_html=True)

# Input grid wrapper
st.markdown('<div class="input-grid">', unsafe_allow_html=True)

input_data = {}

# Create inputs in grid
cols = st.columns(2)
col_idx = 0

for feature in feature_order:
    fl = feature.lower()
    
    with cols[col_idx % 2]:
        if fl == "pregnancies":
            input_data[feature] = st.number_input(
                " Pregnancies",
                min_value=0,
                step=1,
                help="Number of times pregnant",
                key=feature
            )
        elif fl == "glucose":
            input_data[feature] = st.number_input(
                " Glucose Level",
                min_value=0,
                help="Normal: <140 mg/dL",
                key=feature
            )
        elif fl == "bloodpressure":
            input_data[feature] = st.number_input(
                " Blood Pressure",
                min_value=0,
                help="Normal: 80/120 mmHg",
                key=feature
            )
        elif fl == "skinthickness":
            input_data[feature] = st.number_input(
                " Skin Thickness",
                min_value=0,
                help="Triceps skin fold thickness (mm)",
                key=feature
            )
        elif fl == "insulin":
            input_data[feature] = st.number_input(
                " Insulin Level",
                min_value=0,
                help="2-hour serum insulin (μU/mL)",
                key=feature
            )
        elif fl == "bmi":
            input_data[feature] = st.number_input(
                " BMI",
                min_value=0.0,
                format="%.1f",
                help="Normal: 18.5 - 24.9",
                key=feature
            )
        elif fl == "diabetespedigreefunction":
            input_data[feature] = st.number_input(
                " Diabetes Pedigree",
                min_value=0.0,
                format="%.3f",
                help="Family history score",
                key=feature
            )
        elif fl == "age":
            input_data[feature] = st.number_input(
                " Age",
                min_value=0,
                step=1,
                help="Age in years",
                key=feature
            )
    
    col_idx += 1

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button(" Predict Risk", use_container_width=False):
    if pipe is None:
        st.error(" Model not loaded. Please check the model file path.")
    else:
        try:
            # Create DataFrame with proper column order
            input_df = pd.DataFrame([input_data], columns=feature_order)
            
            # Make prediction
            prediction = pipe.predict(input_df)[0]
            probability = pipe.predict_proba(input_df)[0][1]
            
            # Display result with enhanced styling
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-high-risk">
                        <div class="result-status">
                            <span style="font-size:2rem;">⚠️</span>
                            <span style="color:#ff8a80;">High Risk of Diabetes</span>
                        </div>
                        <div class="result-prob">
                            <strong>Probability:</strong> {probability:.2%}
                        </div>
                        <div style="margin-top:12px; color:#889bb8; font-size:0.9rem;">
                            <span> Please consult a healthcare professional for further evaluation.</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-low-risk">
                        <div class="result-status">
                            <span style="font-size:2rem;"></span>
                            <span style="color:#69db7c;">Low Risk of Diabetes</span>
                        </div>
                        <div class="result-prob">
                            <strong>Probability:</strong> {probability:.2%}
                        </div>
                        <div style="margin-top:12px; color:#889bb8; font-size:0.9rem;">
                            <span>💪 Maintain a healthy lifestyle and continue regular checkups.</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card
st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# Footer
st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem 0; color:#374a5e; font-size:0.75rem; letter-spacing:0.3px;">
        <span>🛡️ Powered by Machine Learning · Secure & Confidential</span>
        <br>
        <span style="font-size:0.65rem; opacity:0.6;">v2.0 · Advanced UI</span>
    </div>
""", unsafe_allow_html=True)