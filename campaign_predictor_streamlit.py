import streamlit as st
import numpy as np
import pandas as pd

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="CampaignIQ - AI Marketing Intelligence",
    page_icon="🚀",
    layout="wide"
)

# -------------------------
# Premium Dark UI Styling
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FAFAFA;
}
.stMetric {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
}
button[kind="primary"] {
    height: 55px;
    border-radius: 10px;
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
# Header
# -------------------------
st.title("🚀 CampaignIQ")
st.markdown("""
### AI-Powered Marketing Performance Intelligence Platform  
Real-time campaign success forecasting using weighted performance modeling.
""")

st.divider()

# -------------------------
# Input Section
# -------------------------
st.header("📊 Campaign Configuration")

with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox("Marketing Channel", list(CHANNEL_MULTIPLIERS.keys()))
        acquisition_cost = st.number_input("Acquisition Cost ($)", 5000, 20000, 12500)
        clicks = st.number_input("Total Clicks", 100, 1000, 500)

    with col2:
        conversion_rate = st.number_input("Conversion Rate (decimal)", 0.01, 0.15, 0.08, step=0.01)
        duration = st.number_input("Campaign Duration (days)", 15, 60, 38)
        engagement_score = st.slider("Engagement Score (1–10)", 1.0, 10.0, 5.5, step=0.1)

    predict = st.form_submit_button("🚀 Run AI Prediction")

# -------------------------
# Prediction Logic
# -------------------------
if predict:

    # Deterministic version (removed randomness for thesis stability)
    noise = 0  

    cpc = acquisition_cost / clicks
    channel_score = CHANNEL_MULTIPLIERS[channel]

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
    st.subheader("📈 Executive Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted ROI",
        f"{roi:.2f}",
        delta=f"{roi - TRAINING_MEDIAN:.2f} vs Benchmark"
    )

    col2.metric(
        "Success Probability",
        f"{probability*100:.1f}%"
    )

    col3.metric(
        "Campaign Verdict",
        "SUCCESS" if is_success else "NOT SUCCESS"
    )

    # Probability Level Indicator
    if probability > 0.75:
        st.success("🔥 High Probability Campaign Performance")
    elif probability > 0.55:
        st.warning("⚠ Moderate Campaign Potential")
    else:
        st.error("❌ Low Probability of Success")

    st.divider()

    # -------------------------
    # Contribution Analysis
    # -------------------------
    st.subheader("🧠 AI Contribution Analysis (Explainable Model Output)")

    contributions = {
        "Engagement Impact": engagement_score * 0.8,
        "Channel Strength": channel_score * 10,
        "Conversion Contribution": conversion_rate * 15,
        "Duration Adjustment": -abs(duration - 37.5) * 0.05,
        "Cost Efficiency Impact": cpc * -0.0005
    }

    df = pd.DataFrame({
        "Factor": list(contributions.keys()),
        "Impact Score": list(contributions.values())
    })

    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("Factor"))

    # -------------------------
    # AI Explanation
    # -------------------------
    st.subheader("📘 Model Interpretation Summary")

    if is_success:
        st.write(
            "The model predicts campaign success because positive drivers "
            "(engagement, channel strength, and conversion efficiency) "
            "outweigh cost and duration penalties."
        )
    else:
        st.write(
            "The model predicts underperformance due to insufficient positive "
            "drivers or excessive cost/duration penalties relative to benchmark ROI."
        )

    # -------------------------
    # Download Report
    # -------------------------
    report = f"""
CampaignIQ Executive Report
---------------------------
Channel: {channel}
Predicted ROI: {roi:.2f}
Benchmark ROI: {TRAINING_MEDIAN}
Success Probability: {probability*100:.1f}%
Verdict: {"SUCCESS" if is_success else "NOT SUCCESS"}
"""

    st.download_button(
        "📄 Download Executive Report",
        report,
        file_name="CampaignIQ_Report.txt"
    )
