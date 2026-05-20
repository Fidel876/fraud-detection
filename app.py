# Enhanced Streamlit Fraud Detection Dashboard (`app.py`)
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .title-text {
        font-size: 42px;
        font-weight: 700;
        color: white;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource

def load_model():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2092/2092063.png",
        width=100
    )

    st.title("Fraud AI")

    st.markdown("---")

    st.markdown("### Dashboard Controls")

    show_preview = st.checkbox("Show Dataset Preview", True)
    show_charts = st.checkbox("Show Analytics", True)
    show_fraud_only = st.checkbox("Show Only Fraud Cases", False)

    st.markdown("---")

    st.info(
        "Upload transaction CSV files and analyze fraudulent activity using AI-powered prediction models."
    )

# =====================================
# HEADER
# =====================================
st.markdown(
    '<div class="title-text">🛡️ AI Fraud Detection Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced machine learning powered fraud analysis and transaction monitoring system.</div>',
    unsafe_allow_html=True
)

# =====================================
# FILE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Upload Transaction Dataset",
    type=["csv"]
)

# =====================================
# MAIN PROCESSING
# =====================================
if uploaded_file is not None:

    try:
        # Read data
        data = pd.read_csv(uploaded_file)

        original_rows = len(data)

        # Dataset preview
        if show_preview:
            st.subheader("📄 Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True)

        # Remove Class column if present
        if "Class" in data.columns:
            X = data.drop("Class", axis=1)
        else:
            X = data.copy()

        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Add prediction results
        data["Fraud_Status"] = predictions

        data["Fraud_Status"] = data["Fraud_Status"].map({
            1: "Normal",
            -1: "Fraud"
        })

        # Summary stats
        fraud_count = (data["Fraud_Status"] == "Fraud").sum()
        normal_count = (data["Fraud_Status"] == "Normal").sum()

        fraud_percentage = round((fraud_count / original_rows) * 100, 2)

        # =====================================
        # METRICS
        # =====================================
        st.subheader("📊 Detection Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{original_rows:,}"
            )

        with col2:
            st.metric(
                label="Fraud Cases",
                value=f"{fraud_count:,}"
            )

        with col3:
            st.metric(
                label="Normal Cases",
                value=f"{normal_count:,}"
            )

        with col4:
            st.metric(
                label="Fraud Rate",
                value=f"{fraud_percentage}%"
            )

        # =====================================
        # CHARTS
        # =====================================
        if show_charts:

            st.subheader("📈 Fraud Analytics")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                pie_fig = px.pie(
                    names=["Normal", "Fraud"],
                    values=[normal_count, fraud_count],
                    title="Fraud Distribution",
                    hole=0.45
                )

                pie_fig.update_layout(
                    paper_bgcolor="#0f172a",
                    font_color="white"
                )

                st.plotly_chart(pie_fig, use_container_width=True)

            with chart_col2:
                bar_fig = px.bar(
                    x=["Normal", "Fraud"],
                    y=[normal_count, fraud_count],
                    title="Transaction Classification",
                    text=[normal_count, fraud_count]
                )

                bar_fig.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font_color="white"
                )

                st.plotly_chart(bar_fig, use_container_width=True)

        # =====================================
        # FRAUD TABLE
        # =====================================
        st.subheader("🚨 Fraud Detection Results")

        if show_fraud_only:
            filtered_data = data[data["Fraud_Status"] == "Fraud"]
        else:
            filtered_data = data

        st.dataframe(filtered_data, use_container_width=True)

        # =====================================
        # SUCCESS MESSAGE
        # =====================================
        st.success("Fraud analysis completed successfully.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# =====================================
# EMPTY STATE
# =====================================
else:

    st.markdown("---")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.info(
            "📂 Upload a CSV file to begin fraud detection analysis."
        )

        st.markdown("### Supported Features")

        st.markdown(
            """
            - AI-powered fraud detection
            - Interactive dashboard analytics
            - Fraud transaction filtering
            - Real-time data visualization
            - CSV dataset processing
            """
        )

# =====================================
# FOOTER
# =====================================
st.markdown("---")

st.caption("AI Fraud Detection Dashboard • Built with Streamlit")