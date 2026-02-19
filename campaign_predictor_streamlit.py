import streamlit as st
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Marketing Campaign Success Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# ULTRA STRONG GLOBAL CSS
# -------------------------
st.markdown("""
<style>

/* Force EVERYTHING bigger */
html, body, [class*="css"]  {
    font-size: 26px !important;
}

/* Main title */
h1 {
    font-size: 60px !important;
    font-weight: 900 !important;
}

/* Section headers */
h2 {
    font-size: 42px !important;
}

h3 {
    font-size: 34px !important;
}

/* All labels */
label {
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* Inputs */
input, select, textarea {
    font-size: 24px !important;
}

/* Slider text */
div[data-baseweb="slider"] {
    font-size: 24px !important;
}

/* Buttons */
button[kind="primary"] {
    font-size: 28px !important;
    height: 70px !important;
    border-radius: 14px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    font-size: 30px !important;
}

/* Success/Error */
.stSuccess, .stError {
    font-size: 28px !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
CHANNEL_MULTIPLIERS = {
    'Google Ads': 1.30,
    'YouTube': 1.20,
    'Facebook': 1.10,
    'Instagram': 1.10,
    'Website': 1.05,
    'Email': 1.00,
    'Display': 0.90
}

TRAINING_MEDIAN = 16.08

# -------------------------
# Title
# -------------------------
st.title("📈 Marketing Campaign Success Predictor")
st.markdown("### AI-Based Campaign Performance Estimator")

st.divider()

# -------------------------
# Inputs
# -------------------------
st.header("Campaign Inputs")

with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox("Channel Used", list(CHANNEL_MULTIPLIERS.keys()))
        acquisition_cost = st.number_input("Acquisition Cost ($)", 5000, 20000, 12500)
        clicks = st.number_input("Clicks", 100, 1000, 500)

    with col2:
        conversion_rate = st.number_input("Conversion Rate (decimal)", 0.01, 0.15, 0.08, step=0.01)
        duration = st.number_input("Duration (days)", 15, 60, 38)
        engagement_score = st.slider("Engagement Score (1–10)", 1.0, 10.0, 5.5, step=0.1)

    predict = st.form_submit_button("🚀 Predict Campaign Success")

# -------------------------
# Prediction
# -------------------------
if predict:

    cpc = acquisition_cost / clicks
    channel_score = CHANNEL_MULTIPLIERS[channel]
    noise = np.random.normal(0, 2)

    roi = (
        conversion_rate * 15 +
        cpc * -0.0005 +
        channel_score * 10 +
        -abs(duration - 37.5) * 0.05 +
        engagement_score * 0.8 +
        noise
    )

    probability = 1 / (1 + np.exp(-(roi - TRAINING_MEDIAN)/2))
    is_success = roi > TRAINING_MEDIAN

    st.divider()
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted ROI", f"{roi:.2f}")
    col2.metric("Median ROI", f"{TRAINING_MEDIAN:.2f}")
    col3.metric("Success Probability", f"{probability*100:.1f}%")

    if is_success:
        st.success("✅ Campaign Predicted SUCCESS")
    else:
        st.error("❌ Campaign Predicted NOT SUCCESS")

    st.divider()
    st.subheader("ROI Contribution Breakdown")

    contributions = {
        "Engagement": engagement_score * 0.8,
        "Channel": channel_score * 10,
        "Conversion": conversion_rate * 15,
        "Duration Penalty": -abs(duration - 37.5) * 0.05,
        "CPC Impact": cpc * -0.0005
    }

    st.bar_chart(contributions)
