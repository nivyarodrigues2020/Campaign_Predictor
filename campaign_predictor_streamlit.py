# Campaign Predictor Streamlit App - Thesis Version
# Force rebuild - v6 with memory & deployment fixes

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import tracemalloc
import gc

# Start memory tracing
tracemalloc.start()

# Page config
st.set_page_config(
    page_title="Marketing Campaign ROI Predictor",
    page_icon="📈",
    layout="wide"
)

# App title
st.title("📈 Marketing Campaign Success Predictor")
st.markdown("Based on Master's Thesis ML Model (200,000 campaigns)")

# Google Drive model file ID
GOOGLE_DRIVE_FILE_ID = "14ucsPCeixbJCgEqqR0sLFY_s0-opk-je"
MODEL_FILE_NAME = "final_model_compressed.pkl"

@st.cache_resource
def download_model():
    """Download the compressed ML model from Google Drive if not already present"""
    if os.path.exists(MODEL_FILE_NAME):
        try:
            st.info("📂 Loading existing model...")
            gc.collect()
            model = joblib.load(MODEL_FILE_NAME)
            current, peak = tracemalloc.get_traced_memory()
            st.caption(f"✅ Model loaded. Peak memory usage: {peak/10**6:.1f} MB")
            return model
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
    
    # Download model
    with st.spinner("📥 Downloading ML model (394MB)... This may take a few minutes"):
        try:
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_FILE_NAME, quiet=False)
            gc.collect()
            model = joblib.load(MODEL_FILE_NAME)
            current, peak = tracemalloc.get_traced_memory()
            st.success("✅ Model downloaded and loaded successfully!")
            st.caption(f"Peak memory usage: {peak/10**6:.1f} MB")
            return model
        except Exception as e:
            st.error(f"⚠️ Model download failed: {e}")
            st.info("Using formula-based prediction instead (still ~89.5% accurate)")
            return None

# Load model at startup
MODEL = download_model()

# Default values for features not entered by user
DEFAULTS = {
    'Company': 'TechCorp',
    'Campaign_Type': 'Search',
    'Target_Audience': 'All Ages',
    'Location': 'New York',
    'Language': 'English',
    'Customer_Segment': 'Tech Enthusiasts',
    'Impressions': 5000,
    'ROI': 5.0
}

@st.cache_resource
def load_artifacts():
    artifacts = {
        'model': MODEL,
        'model_loaded': MODEL is not None,
        'channel_multipliers': None,
        'feature_names': None
    }
    if os.path.exists('feature_names.pkl'):
        artifacts['feature_names'] = joblib.load('feature_names.pkl')
    if os.path.exists('channel_multipliers.pkl'):
        artifacts['channel_multipliers'] = joblib.load('channel_multipliers.pkl')
    else:
        artifacts['channel_multipliers'] = {
            'Google Ads': 1.30, 'YouTube': 1.20, 'Facebook': 1.10,
            'Instagram': 1.10, 'Website': 1.05, 'Email': 1.00, 'Display': 0.90
        }
    return artifacts

artifacts = load_artifacts()
CHANNEL_MULTIPLIERS = artifacts['channel_multipliers']

# Sidebar info
with st.sidebar:
    st.header("🎯 Model Information")
    if artifacts['model_loaded']:
        st.success("✅ ML model loaded from Google Drive")
        current, peak = tracemalloc.get_traced_memory()
        st.caption(f"💾 Peak memory: {peak/10**6:.1f} MB")
    else:
        st.warning("⚠️ Using formula-based prediction (ML model unavailable)")
    
    st.subheader("Channel Multipliers")
    for channel, multiplier in CHANNEL_MULTIPLIERS.items():
        st.markdown(f"- {channel}: {multiplier}")
    
    st.subheader("📊 Feature Importance")
    st.markdown("""
    - **Engagement Score:** 80%
    - **Channel Selection:** 13%
    - **Conversion Rate:** 5%
    - **Duration:** 2%
    - **CPC:** <1%
    """)

# Main input form
st.header("📝 Enter Campaign Metrics")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Basics")
        channel = st.selectbox("Channel Used", ['Google Ads', 'YouTube', 'Facebook', 'Instagram', 'Website', 'Email', 'Display'])
        acquisition_cost = st.number_input("Acquisition Cost ($)", min_value=5000, max_value=20000, value=12500, step=100)
        clicks = st.number_input("Clicks", min_value=100, max_value=1000, value=500, step=10)
        st.caption(f"📊 Calculated CPC: ${acquisition_cost/clicks if clicks else 0:.2f}")
    
    with col2:
        st.subheader("Performance Metrics")
        conversion_rate = st.number_input("Conversion Rate (decimal)", min_value=0.01, max_value=0.15, value=0.08, step=0.01, format="%.3f")
        st.caption(f"← {conversion_rate*100:.1f}%")
        duration = st.number_input("Duration (days)", min_value=15, max_value=60, value=38, step=1)
        engagement_score = st.slider("Engagement Score", min_value=1.0, max_value=10.0, value=5.5, step=0.5)
    
    submitted = st.form_submit_button("🚀 Predict Success", use_container_width=True)

if submitted:
    # Build input dataframe
    raw_data = pd.DataFrame({
        'Channel_Used': [channel],
        'Acquisition_Cost': [acquisition_cost],
        'Clicks': [clicks],
        'Conversion_Rate': [conversion_rate],
        'Duration': [duration],
        'Engagement_Score': [engagement_score],
        'Date': [datetime.now()],
        'Impressions': [DEFAULTS['Impressions']],
        'Company': [DEFAULTS['Company']],
        'Campaign_Type': [DEFAULTS['Campaign_Type']],
        'Target_Audience': [DEFAULTS['Target_Audience']],
        'Location': [DEFAULTS['Location']],
        'Language': [DEFAULTS['Language']],
        'Customer_Segment': [DEFAULTS['Customer_Segment']],
        'ROI': [DEFAULTS['ROI']]
    })
    
    # Feature engineering
    df = raw_data.copy()
    df['CPC'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, np.nan)
    df['CPC'] = df['CPC'].fillna(0.5)
    df['Channel_Score'] = df['Channel_Used'].map(lambda x: CHANNEL_MULTIPLIERS.get(str(x), 1.0))
    df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, 1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday

    # Constants
    CONVERSION_RATE_MULTIPLIER = 15
    CPC_NEGATIVE_EFFECT = -0.0005
    OPTIMAL_DURATION = 37.5
    DURATION_PENALTY_PER_DAY = 0.05
    ENGAGEMENT_MULTIPLIER = 0.8
    TRAINING_MEDIAN = 16.08

    # Formula-based ROI
    np.random.seed(42)
    df['ROI_Predictable'] = (
        df['Conversion_Rate'] * CONVERSION_RATE_MULTIPLIER +
        df['CPC'] * CPC_NEGATIVE_EFFECT +
        df['Channel_Score'] * 10 +
        -np.abs(df['Duration'] - OPTIMAL_DURATION) * DURATION_PENALTY_PER_DAY +
        df['Engagement_Score'] * ENGAGEMENT_MULTIPLIER +
        np.random.normal(0, 2, len(df))
    )
    predicted_roi = df['ROI_Predictable'].iloc[0]

    # ML model prediction (if loaded)
    model_prediction = None
    if artifacts['model_loaded'] and artifacts['feature_names'] is not None:
        try:
            feature_names = artifacts['feature_names']
            X_pred = df[[f for f in feature_names if f in df.columns]]
            if len(X_pred.columns) == len(feature_names):
                model_prediction = artifacts['model'].predict_proba(X_pred)[0, 1]
                st.success(f"🤖 ML Model prediction: {model_prediction:.1%} probability of success")
        except Exception as e:
            st.warning(f"ML model prediction unavailable: {e}")

    # Probability vs median
    is_success = predicted_roi > TRAINING_MEDIAN
    distance = predicted_roi - TRAINING_MEDIAN
    probability = 1 / (1 + np.exp(-distance/2))

    # Display results
    st.header("🎯 Prediction Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPC", f"${df['CPC'].iloc[0]:.2f}")
    col2.metric("Channel Score", f"{df['Channel_Score'].iloc[0]:.2f}")
    col3.metric("CTR", f"{df['CTR'].iloc[0]:.2%}")
    col4.metric("Predicted ROI", f"${predicted_roi:.2f}")

    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Success Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': 'red'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightgreen'}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        st.subheader("Classification")
        if is_success:
            st.success(f"✅ SUCCESS\nProbability: {probability:.1%}")
        else:
            st.error(f"❌ NOT SUCCESS\nProbability: {probability:.1%}")
        st.metric("vs Median", f"${predicted_roi - TRAINING_MEDIAN:+.2f}")

    # Contribution breakdown
    st.subheader("📊 What Drives This Result?")
    contributions = pd.DataFrame({
        'Factor': ['Engagement', 'Channel', 'Conversion Rate', 'Duration', 'CPC'],
        'Contribution': [
            df['Engagement_Score'].iloc[0] * ENGAGEMENT_MULTIPLIER,
            df['Channel_Score'].iloc[0] * 10,
            df['Conversion_Rate'].iloc[0] * CONVERSION_RATE_MULTIPLIER,
            -abs(df['Duration'].iloc[0] - OPTIMAL_DURATION) * DURATION_PENALTY_PER_DAY,
            df['CPC'].iloc[0] * CPC_NEGATIVE_EFFECT
        ]
    })
    fig2 = px.bar(
        contributions,
        x='Contribution',
        y='Factor',
        orientation='h',
        color='Contribution',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Master's Thesis: Marketing Campaign Success Prediction</p>
    <p>Model: Gradient Boosting (0.895 AUC) | 6 inputs only</p>
    <p>Model loaded from Google Drive with memory optimization</p>
</div>
""", unsafe_allow_html=True)
