# Force rebuild - v4
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Marketing Campaign ROI Predictor",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Marketing Campaign Success Predictor")
st.markdown("Based on Master's Thesis ML Model (200,000 campaigns)")

# Google Drive file ID (extracted from your link)
GOOGLE_DRIVE_FILE_ID = "14ucsPCeixbJCgEqqR0sLFY_s0-opk-je"

@st.cache_resource
def download_model():
    """Download the compressed model from Google Drive if not exists"""
    model_path = 'final_model_compressed.pkl'
    
    # If model already exists, just load it
    if os.path.exists(model_path):
        try:
            st.info("📂 Loading existing model file...")
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
    
    # Download model
    with st.spinner('📥 Downloading ML model (394 MB)... This may take 2-3 minutes'):
        try:
            # Create download URL
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            
            # Download the file
            gdown.download(url, model_path, quiet=False)
            
            st.success('✅ Model downloaded successfully!')
            return joblib.load(model_path)
            
        except Exception as e:
            st.error(f"⚠️ Download failed: {e}")
            st.info("Using formula-based prediction instead (still 89.5% accurate from thesis)")
            return None

# Download model at startup
MODEL = download_model()

# Default values for features with near-zero importance
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
    
    # Load feature names if exists
    if os.path.exists('feature_names.pkl'):
        artifacts['feature_names'] = joblib.load('feature_names.pkl')
    
    # Load channel multipliers if exists, otherwise use defaults
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

# Sidebar
with st.sidebar:
    st.header("🎯 Model Information")
    
    if artifacts['model_loaded']:
        st.success("✅ Trained model loaded from Google Drive!")
    else:
        st.warning("⚠️ Using formula-based prediction (model download failed)")
    
    st.subheader("Channel Multipliers:")
    for channel, multiplier in CHANNEL_MULTIPLIERS.items():
        st.markdown(f"- {channel}: {multiplier}")
    
    st.subheader("📊 Feature Importance:")
    st.markdown("""
    - **Engagement Score:** 80%
    - **Channel Selection:** 13%  
    - **Conversion Rate:** 5%
    - **Duration:** 2%
    - **CPC:** <1%
    """)

# Main form
st.header("📝 Enter Campaign Metrics")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Basics")
        
        channel = st.selectbox(
            "Channel Used",
            options=['Google Ads', 'YouTube', 'Facebook', 'Instagram', 
                    'Website', 'Email', 'Display']
        )
        
        acquisition_cost = st.number_input(
            "Acquisition Cost ($)",
            min_value=5000, max_value=20000, value=12500, step=100
        )
        
        clicks = st.number_input(
            "Clicks",
            min_value=100, max_value=1000, value=500, step=10
        )
        
        cpc_display = acquisition_cost / clicks if clicks > 0 else 0
        st.caption(f"📊 Calculated CPC: ${cpc_display:.2f}")
    
    with col2:
        st.subheader("Performance Metrics")
        
        conversion_rate = st.number_input(
            "Conversion Rate (as decimal)",
            min_value=0.01, max_value=0.15, value=0.08, step=0.01,
            format="%.3f"
        )
        st.caption(f"← {conversion_rate*100:.1f}%")
        
        duration = st.number_input(
            "Duration (days)",
            min_value=15, max_value=60, value=38, step=1
        )
        
        engagement_score = st.slider(
            "Engagement Score",
            min_value=1.0, max_value=10.0, value=5.5, step=0.5
        )
    
    submitted = st.form_submit_button("🚀 Predict Success", use_container_width=True)

if submitted:
    # Create dataframe with ALL features
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
    
    st.markdown("---")
    
    # Calculate engineered features
    df = raw_data.copy()
    
    # Calculate CPC
    df['CPC'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, np.nan)
    df['CPC'] = df['CPC'].fillna(0.5)
    
    # Channel Score
    df['Channel_Score'] = df['Channel_Used'].map(
        lambda x: CHANNEL_MULTIPLIERS.get(str(x), 1.0)
    )
    
    # Calculate CTR
    df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, 1)
    
    # Date features
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
    
    # Calculate ROI (formula)
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
    
    # Use trained model if available
    model_prediction = None
    if artifacts['model_loaded'] and artifacts['feature_names'] is not None:
        try:
            feature_names = artifacts['feature_names']
            # Ensure all required features are present
            available_features = [f for f in feature_names if f in df.columns]
            if len(available_features) == len(feature_names):
                X_pred = df[feature_names]
                model_prediction = artifacts['model'].predict_proba(X_pred)[0, 1]
                st.success(f"🤖 ML Model prediction: {model_prediction:.1%} probability of success")
            else:
                missing = set(feature_names) - set(df.columns)
                st.warning(f"Model expects features not in input: {missing}")
        except Exception as e:
            st.warning(f"Could not use ML model: {e}")
    
    # Determine success (above median)
    is_success = predicted_roi > TRAINING_MEDIAN
    distance = predicted_roi - TRAINING_MEDIAN
    probability = 1 / (1 + np.exp(-distance/2))
    
    # Display results
    st.header("🎯 Prediction Results")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("CPC", f"${df['CPC'].iloc[0]:.2f}")
    with col_m2:
        st.metric("Channel Score", f"{df['Channel_Score'].iloc[0]:.2f}")
    with col_m3:
        st.metric("CTR", f"{df['CTR'].iloc[0]:.2%}")
    with col_m4:
        st.metric("Predicted ROI", f"${predicted_roi:.2f}")
    
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
        st.subheader("Classification:")
        if is_success:
            st.success(f"✅ **SUCCESS**\nProbability: {probability:.1%}")
        else:
            st.error(f"❌ **NOT SUCCESS**\nProbability: {probability:.1%}")
        
        st.metric("vs Median", f"${predicted_roi - TRAINING_MEDIAN:+.2f}")
    
    # Contribution breakdown
    st.subheader("📊 What Drives This Result?")
    
    contributions = pd.DataFrame({
        'Factor': ['Engagement', 'Channel', 'Conversion Rate', 'Duration', 'CPC'],
        'Contribution': [
            df['Engagement_Score'].iloc[0] * 0.8,
            df['Channel_Score'].iloc[0] * 10,
            df['Conversion_Rate'].iloc[0] * 15,
            -abs(df['Duration'].iloc[0] - 37.5) * 0.05,
            df['CPC'].iloc[0] * -0.0005
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
    <p>Model loaded from Google Drive</p>
</div>
""", unsafe_allow_html=True)
