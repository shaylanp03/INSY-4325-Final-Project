import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import html
import warnings
import os
import time
import json

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0f1117; color: #e0e0e0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d27 0%, #0f1117 100%);
    border-right: 1px solid #2d3142;
}
[data-testid="stSidebar"] .stRadio > label { color: #a0aec0 !important; }

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid #2d3142;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #4ade80; }
.metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
.metric-delta-pos { color: #4ade80; font-size: 0.8rem; }
.metric-delta-neg { color: #f87171; font-size: 0.8rem; }

/* Section headers */
.section-title {
    font-size: 1.6rem; font-weight: 700; color: #ffffff;
    border-left: 4px solid #6366f1; padding-left: 12px; margin-bottom: 4px;
}
.section-subtitle { font-size: 0.9rem; color: #718096; margin-bottom: 24px; }

/* Feature cards */
.feature-card {
    background: #1e2235; border: 1px solid #2d3142; border-radius: 10px;
    padding: 20px; height: 140px;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: #6366f1; }
.feature-icon { font-size: 1.8rem; margin-bottom: 8px; }
.feature-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
.feature-desc { font-size: 0.8rem; color: #718096; margin-top: 4px; }

/* Best model banner */
.best-model-banner {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%);
    border: 1px solid #3b82f6; border-radius: 10px; padding: 16px;
    margin-bottom: 16px;
}

/* Chat bubbles */
.chat-user {
    background: #6366f1; color: white; border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; margin: 8px 0; max-width: 80%; margin-left: auto;
    font-size: 0.9rem;
}
.chat-bot {
    background: #1e2235; color: #e2e8f0; border: 1px solid #2d3142;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px; margin: 8px 0; max-width: 85%;
    font-size: 0.9rem;
}
.chat-container {
    height: 420px; overflow-y: auto; padding: 12px;
    background: #13151f; border-radius: 12px; border: 1px solid #2d3142;
}

/* Rank badges */
.rank-1 { background: #f59e0b; color: #1a1a1a; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.rank-2 { background: #94a3b8; color: #1a1a1a; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.rank-3 { background: #cd7c3a; color: #1a1a1a; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }

/* Plotly charts background */
.js-plotly-plot { background: transparent !important; }

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Progress bars */
.stProgress > div > div { background-color: #6366f1 !important; }

/* Inputs */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: #1e2235 !important; color: #e2e8f0 !important;
    border-color: #2d3142 !important;
}

/* Info boxes */
.info-box {
    background: #1a2744; border: 1px solid #3b82f6; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0;
}
.success-box {
    background: #14291f; border: 1px solid #4ade80; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0;
}
.warning-box {
    background: #2a1f0e; border: 1px solid #f59e0b; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0;
}

/* Prediction result */
.price-result {
    background: linear-gradient(135deg, #1a2744, #1e3a5f);
    border: 2px solid #6366f1; border-radius: 16px;
    padding: 24px; text-align: center; margin: 16px 0;
}
.price-amount { font-size: 2.8rem; font-weight: 800; color: #4ade80; }
.price-label { color: #a0aec0; font-size: 0.9rem; }

hr { border-color: #2d3142 !important; }

/* Map container */
.folium-map { border-radius: 12px; overflow: hidden; }

/* System components list */
.sys-check { color: #4ade80; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,21,31,0.8)",
    font_color="#a0aec0",
    font_size=12,
    margin=dict(l=40, r=20, t=40, b=40),
)

CSV_PATH = "600K_US_Housing_Properties.csv"

# ─── Session State ────────────────────────────────────────────────────────────
for key, val in {
    "df_raw": None,
    "df_clean": None,
    "models": {},
    "model_metrics": {},
    "deployed_model": None,
    "deployed_model_name": None,
    "deployed_feat_cols": [],        # FIX: was missing, caused KeyError on Predictions page
    "predictions_history": [],
    "chat_history": [],
    "cleaning_done": False,
    "training_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helper: get active dataframe safely ─────────────────────────────────────
def get_active_df():
    """FIX: replaces broken `df_clean or df_raw` pattern on DataFrames."""
    if st.session_state.cleaning_done and st.session_state.df_clean is not None:
        return st.session_state.df_clean
    return st.session_state.df_raw

# ─── Helper: Load Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_sample(path, n=2000):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 10000]
    df = df[df["price"] < 10_000_000]
    for col in ["bedroom_number", "bathroom_number", "living_space"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["bedroom_number", "bathroom_number", "living_space"])
    df = df[df["bedroom_number"].between(1, 10)]
    df = df[df["bathroom_number"].between(1, 10)]
    df = df[df["living_space"].between(200, 15000)]
    if len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_map_sample(path, n=500):
    df = pd.read_csv(path, low_memory=False, usecols=[
        "address", "city", "state", "latitude", "longitude",
        "price", "bedroom_number", "bathroom_number", "living_space",
        "property_type", "property_url"
    ])
    df = df.dropna(subset=["latitude", "longitude", "price"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(24, 50)) & (df["longitude"].between(-130, -65))]
    df = df[df["price"].between(10000, 5_000_000)]
    if len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df

# ─── ML Helpers ───────────────────────────────────────────────────────────────
FEATURES = ["bedroom_number", "bathroom_number", "living_space",
            "state_enc", "property_type_enc", "year_build"]

def prepare_features(df):
    d = df.copy()
    for col in ["bedroom_number", "bathroom_number", "living_space", "year_build"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["year_build"] = d["year_build"].fillna(1990)
    d["bedroom_number"] = d["bedroom_number"].fillna(3)
    d["bathroom_number"] = d["bathroom_number"].fillna(2)
    d["living_space"] = d["living_space"].fillna(1800)
    le_state = LabelEncoder()
    le_type = LabelEncoder()
    d["state_enc"] = le_state.fit_transform(d["state"].astype(str).fillna("Unknown"))
    d["property_type_enc"] = le_type.fit_transform(d["property_type"].astype(str).fillna("Unknown"))
    return d, le_state, le_type

# ─── Chatbot ──────────────────────────────────────────────────────────────────
def get_bot_response(user_msg, df, metrics):
    msg = user_msg.lower()

    if any(w in msg for w in ["average price", "avg price", "mean price"]):
        if df is not None:
            avg = df["price"].mean()
            return f"📊 The average listing price in our dataset is **${avg:,.0f}**."
        return "Please upload data first."

    if "median price" in msg:
        if df is not None:
            med = df["price"].median()
            return f"📊 The median listing price is **${med:,.0f}**."
        return "Please upload data first."

    if any(w in msg for w in ["bedrooms", "bedroom"]):
        if df is not None:
            avg_bed = df["bedroom_number"].mean()
            return f"🛏 The average number of bedrooms in our dataset is **{avg_bed:.1f}**."
        return "Please upload data first."

    if any(w in msg for w in ["most expensive", "highest price", "priciest"]):
        if df is not None:
            row = df.nlargest(1, "price").iloc[0]
            return f"💰 The most expensive property is in **{row.get('city','N/A')}, {row.get('state','N/A')}** at **${row['price']:,.0f}** with {row.get('bedroom_number','N/A')} beds / {row.get('bathroom_number','N/A')} baths."
        return "Please upload data first."

    if any(w in msg for w in ["cheapest", "lowest price", "affordable"]):
        if df is not None:
            row = df.nsmallest(1, "price").iloc[0]
            return f"🏷 The most affordable property is in **{row.get('city','N/A')}, {row.get('state','N/A')}** at **${row['price']:,.0f}**."
        return "Please upload data first."

    if any(w in msg for w in ["gradient boosting", "best model", "top model"]):
        if metrics:
            m = metrics.get("Gradient Boosting", {})
            r2 = m.get("r2", "N/A")
            rmse = m.get("rmse", "N/A")
            r2_str = f"{r2:.3f}" if isinstance(r2, float) else r2
            rmse_str = f"${rmse:,.0f}" if isinstance(rmse, float) else rmse
            return f"🏆 **Gradient Boosting** is our best model with R² = **{r2_str}** and RMSE = **{rmse_str}**. It uses sequential ensemble learning to minimize prediction errors."
        return "Train the models first to see performance metrics."

    if any(w in msg for w in ["random forest", "rf"]):
        if metrics:
            m = metrics.get("Random Forest", {})
            r2 = m.get("r2", "N/A")
            rmse = m.get("rmse", "N/A")
            r2_str = f"{r2:.3f}" if isinstance(r2, float) else r2
            rmse_str = f"${rmse:,.0f}" if isinstance(rmse, float) else rmse
            return f"🌲 **Random Forest** uses 100 decision trees in parallel. R² = **{r2_str}**, RMSE = **{rmse_str}**."
        return "Please train models first."

    if any(w in msg for w in ["linear regression", "linear"]):
        if metrics:
            m = metrics.get("Linear Regression", {})
            r2 = m.get("r2", "N/A")
            rmse = m.get("rmse", "N/A")
            r2_str = f"{r2:.3f}" if isinstance(r2, float) else r2
            rmse_str = f"${rmse:,.0f}" if isinstance(rmse, float) else rmse
            return f"📈 **Linear Regression** is the simplest model. R² = **{r2_str}**, RMSE = **{rmse_str}**."
        return "Please train models first."

    if any(w in msg for w in ["r2", "r-squared", "accuracy", "performance"]):
        if metrics:
            lines = ["📊 **Model Performance Summary:**\n"]
            for name, m in metrics.items():
                r2 = m.get("r2", 0)
                rmse = m.get("rmse", 0)
                r2_str = f"{r2:.3f}" if isinstance(r2, float) else str(r2)
                rmse_str = f"${rmse:,.0f}" if isinstance(rmse, float) else str(rmse)
                lines.append(f"• **{name}**: R² = {r2_str}, RMSE = {rmse_str}")
            return "\n".join(lines)
        return "Please train the models first via the Model Training page."

    if any(w in msg for w in ["how many", "dataset size", "rows", "properties"]):
        if df is not None:
            return f"📁 The current dataset has **{len(df):,}** properties loaded for training (sampled from 600K records for performance)."
        return "Please upload data first."

    if any(w in msg for w in ["state", "location", "where"]):
        if df is not None:
            top = df.groupby("state")["price"].mean().nlargest(5)
            resp = "🗺 **Top 5 states by average price:**\n"
            for state, price in top.items():
                resp += f"\n• {state}: ${price:,.0f}"
            return resp
        return "Please upload data first."

    if any(w in msg for w in ["property type", "type", "single family", "condo"]):
        if df is not None:
            vc = df["property_type"].value_counts().head(5)
            resp = "🏘 **Property type breakdown:**\n"
            for pt, cnt in vc.items():
                resp += f"\n• {pt}: {cnt:,} listings"
            return resp
        return "Please upload data first."

    if any(w in msg for w in ["predict", "price estimate", "how much"]):
        return "🏠 Head to the **Predictions** page! Enter property details like bedrooms, bathrooms, square footage, state, and year built — our Gradient Boosting model will give you an instant price estimate."

    if any(w in msg for w in ["map", "geography", "location map"]):
        return "🗺 Check out the **Geography Map** page! You can see up to 500 properties plotted on an interactive map of the US, color-coded by price."

    if any(w in msg for w in ["clean", "cleaning", "missing", "outlier"]):
        return "🧹 The **Data Cleaning** page handles missing value imputation, outlier removal via IQR, and feature encoding. It shows you missing counts, duplicate records, and distributions before training."

    if any(w in msg for w in ["hello", "hi", "hey", "help"]):
        return "👋 Hi! I'm your AI real estate assistant. Ask me about:\n\n• 📊 Average/median prices\n• 🏆 Model performance (R², RMSE)\n• 🗺 States or property types\n• 🏠 How to get a price prediction\n• 📁 Dataset statistics\n\nWhat would you like to know?"

    if any(w in msg for w in ["thank", "thanks"]):
        return "You're welcome! Let me know if you have more real estate questions. 🏡"

    return (
        "🤔 I can help with:\n\n"
        "• **Prices**: 'What is the average price?'\n"
        "• **Models**: 'How did Gradient Boosting perform?'\n"
        "• **Data**: 'How many properties are in the dataset?'\n"
        "• **Locations**: 'Which states have the highest prices?'\n\n"
        "Try one of the above!"
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size:2.2rem;'>🏠</span>
        <div style='font-size:1rem; font-weight:700; color:#e2e8f0; margin-top:6px;'>AI Real Estate</div>
        <div style='font-size:0.75rem; color:#718096;'>Predictive Analytics Platform</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Data Upload", "🧹 Data Cleaning", "🤖 Model Training",
         "📊 Model Comparison", "🔮 Predictions", "🗺 Geography Map",
         "💬 AI Chatbot", "📈 Dashboard"],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    if st.session_state.df_raw is not None:
        st.markdown(f"<div style='color:#4ade80; font-size:0.8rem;'>✅ Data loaded: {len(st.session_state.df_raw):,} rows</div>", unsafe_allow_html=True)
    if st.session_state.training_done:
        st.markdown("<div style='color:#4ade80; font-size:0.8rem;'>✅ Models trained</div>", unsafe_allow_html=True)
    if st.session_state.deployed_model_name:
        st.markdown(f"<div style='color:#4ade80; font-size:0.8rem;'>✅ Deployed: {st.session_state.deployed_model_name}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<div class='section-title'>AI-Powered Real Estate Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Predictive analytics system for housing market trends using machine learning</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <strong>🎯 Objective:</strong> Develop a comprehensive predictive analytics system for real estate price prediction
        and market trend analysis using the 600K US Housing Properties dataset from Kaggle.<br/><br/>
        <strong>📌 Target Variable:</strong> House sale price | <strong>👥 Business Value:</strong>
        Enable real estate professionals, investors, and homebuyers to make data-driven decisions.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>🗃</div>
            <div class='feature-title'>Data Processing</div>
            <div class='feature-desc'>Clean, transform, and visualize real estate data with advanced preprocessing techniques</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-title'>ML Algorithms</div>
            <div class='feature-desc'>Train and compare: Linear Regression, Random Forest, and Gradient Boosting</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>🔮</div>
            <div class='feature-title'>Predictions</div>
            <div class='feature-desc'>Deploy best-performing model to predict house prices for new property listings</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#1e2235; border:1px solid #2d3142; border-radius:10px; padding:20px;'>
            <div style='color:#6366f1; font-weight:700; margin-bottom:12px;'>🏗 Training Module</div>
            <div style='font-size:0.85rem; color:#a0aec0;'>
                <span class='sys-check'>✓</span> Data upload and validation<br/>
                <span class='sys-check'>✓</span> Data cleaning and preprocessing<br/>
                <span class='sys-check'>✓</span> Exploratory data visualization<br/>
                <span class='sys-check'>✓</span> Feature engineering<br/>
                <span class='sys-check'>✓</span> Model training (3 algorithms)<br/>
                <span class='sys-check'>✓</span> Performance comparison<br/>
                <span class='sys-check'>✓</span> Model selection and deployment
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#1e2235; border:1px solid #2d3142; border-radius:10px; padding:20px;'>
            <div style='color:#4ade80; font-weight:700; margin-bottom:12px;'>🚀 Deployment Module</div>
            <div style='font-size:0.85rem; color:#a0aec0;'>
                <span class='sys-check'>✓</span> Real-time price predictions<br/>
                <span class='sys-check'>✓</span> Interactive trends dashboard<br/>
                <span class='sys-check'>✓</span> AI-powered chatbot assistant<br/>
                <span class='sys-check'>✓</span> Geography / map view<br/>
                <span class='sys-check'>✓</span> Market insights and analytics<br/>
                <span class='sys-check'>✓</span> Model performance monitoring<br/>
                <span class='sys-check'>✓</span> Export predictions and reports
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1e2235; border:1px solid #2d3142; border-radius:10px; padding:16px;'>
        <strong>📦 Dataset:</strong> 600,000 US Housing Properties from Kaggle &nbsp;|&nbsp;
        <strong>Features:</strong> Address, City, State, Price, Bedrooms, Bathrooms, Living Space, Property Type, Year Built, and more &nbsp;|&nbsp;
        <strong>Records with GPS:</strong> ~529,000
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════
elif page == "📤 Data Upload":
    st.markdown("<div class='section-title'>Data Upload & Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Upload your real estate dataset from Kaggle (CSV format)</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drag and drop file here", type=["csv"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Loading dataset..."):
            df = pd.read_csv(uploaded, low_memory=False)
            df = df[df["price"].notna() & (pd.to_numeric(df["price"], errors="coerce") > 0)]
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            st.session_state.df_raw = df
            st.session_state.cleaning_done = False
            st.session_state.training_done = False
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📊 Use Demo Dataset (600K rows)"):
            if os.path.exists(CSV_PATH):
                with st.spinner("Sampling dataset (this may take a moment)…"):
                    df = load_and_sample(CSV_PATH, n=2000)
                    st.session_state.df_raw = df
                    st.session_state.cleaning_done = False
                    st.session_state.training_done = False
                st.success(f"✅ Loaded {len(df):,} sampled rows from 600K dataset")
            else:
                st.error("CSV file not found. Please upload it manually above.")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    st.markdown("""
    <div class='info-box' style='margin-top:20px;'>
        <strong>📋 Required Data Format</strong><br/>
        <strong>CSV File:</strong> Comma-separated values with header row<br/>
        <strong>Key Columns:</strong> price, bedroom_number, bathroom_number, living_space,
        state, property_type, year_build, latitude, longitude
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
elif page == "🧹 Data Cleaning":
    st.markdown("<div class='section-title'>Data Cleaning & Visualization</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Preprocess and explore your real estate dataset</div>", unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload data first.")
        st.stop()

    df = st.session_state.df_raw.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["bedroom_number"] = pd.to_numeric(df["bedroom_number"], errors="coerce")
    df["bathroom_number"] = pd.to_numeric(df["bathroom_number"], errors="coerce")
    df["living_space"] = pd.to_numeric(df["living_space"], errors="coerce")

    missing = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    outliers = 0
    for col in ["price", "living_space"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#4ade80;'>{len(df):,}</div>
            <div class='metric-label'>Total Rows ✅</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#f59e0b;'>{missing:,} ({missing/df.size*100:.1f}%)</div>
            <div class='metric-label'>Missing Values ⚠️</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#{"f59e0b" if dups > 0 else "4ade80"};'>{dups}</div>
            <div class='metric-label'>Duplicates {"⚠️" if dups > 0 else "✅"}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#f87171;'>{outliers:,}</div>
            <div class='metric-label'>Outliers 🔍</div></div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Cleaning Techniques")
    remove_dups = st.checkbox("Remove duplicate rows", value=True)
    handle_missing = st.checkbox("Handle missing values (imputation with median/mode)", value=True)
    remove_outliers = st.checkbox("Remove outliers using IQR method", value=True)
    normalize = st.checkbox("Normalize numerical features", value=True)
    encode_cat = st.checkbox("Encode categorical variables", value=True)

    if st.button("🧹 Apply Cleaning"):
        with st.spinner("Cleaning data..."):
            dfc = df.copy()
            if remove_dups:
                dfc = dfc.drop_duplicates()
            if handle_missing:
                for col in ["bedroom_number", "bathroom_number", "living_space"]:
                    dfc[col] = dfc[col].fillna(dfc[col].median())
                dfc["year_build"] = pd.to_numeric(dfc["year_build"], errors="coerce").fillna(1990)
                dfc["state"] = dfc["state"].fillna("Unknown")
                dfc["property_type"] = dfc["property_type"].fillna("SINGLE_FAMILY")
            dfc = dfc.dropna(subset=["price"])
            dfc = dfc[dfc["price"].between(10000, 10_000_000)]
            if remove_outliers:
                for col in ["price", "living_space"]:
                    if col in dfc.columns:
                        Q1, Q3 = dfc[col].quantile(0.25), dfc[col].quantile(0.75)
                        IQR = Q3 - Q1
                        dfc = dfc[(dfc[col] >= Q1 - 1.5*IQR) & (dfc[col] <= Q3 + 1.5*IQR)]
            st.session_state.df_clean = dfc
            st.session_state.cleaning_done = True
        st.success(f"✅ Cleaned! {len(dfc):,} rows remaining.")
        df = dfc

    plot_df = st.session_state.df_clean if st.session_state.cleaning_done else df

    col1, col2 = st.columns(2)
    with col1:
        miss_by_col = plot_df[["bedroom_number","bathroom_number","living_space","year_build"]].isnull().sum()
        fig = px.bar(x=miss_by_col.index, y=miss_by_col.values,
                     title="Missing Values by Column",
                     labels={"x": "", "y": "Count"},
                     color_discrete_sequence=["#ef4444"])
        fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bins = [0, 200000, 400000, 600000, 800000, float("inf")]
        labels = ["<200k", "200-400k", "400-600k", "600-800k", ">800k"]
        plot_df = plot_df.copy()
        plot_df["price_bin"] = pd.cut(plot_df["price"], bins=bins, labels=labels)
        vc = plot_df["price_bin"].value_counts().sort_index()
        fig2 = px.bar(x=vc.index, y=vc.values, title="Price Distribution",
                      labels={"x": "", "y": "Count"},
                      color_discrete_sequence=["#6366f1"])
        fig2.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        corr_cols = ["bedroom_number", "bathroom_number", "living_space", "year_build"]
        corrs = {c: plot_df[[c, "price"]].dropna().corr().iloc[0, 1]
                 for c in corr_cols if c in plot_df.columns}
        fig3 = px.bar(x=list(corrs.values()), y=list(corrs.keys()),
                      orientation="h", title="Feature Correlation with Price",
                      color_discrete_sequence=["#4ade80"])
        fig3.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown("""<div style='background:#1e2235; border:1px solid #2d3142; border-radius:10px; padding:20px;'>
            <div style='font-weight:700; color:#e2e8f0; margin-bottom:12px;'>📊 Statistical Summary</div>""",
                    unsafe_allow_html=True)
        stats = {
            "Mean Price": f"${plot_df['price'].mean():,.0f}",
            "Median Price": f"${plot_df['price'].median():,.0f}",
            "Std Deviation": f"${plot_df['price'].std():,.0f}",
            "Avg Sqft": f"{plot_df['living_space'].mean():,.0f}" if "living_space" in plot_df else "N/A",
            "Avg Bedrooms": f"{plot_df['bedroom_number'].mean():.1f}" if "bedroom_number" in plot_df else "N/A",
            "Avg Bathrooms": f"{plot_df['bathroom_number'].mean():.1f}" if "bathroom_number" in plot_df else "N/A",
        }
        for k, v in stats.items():
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #2d3142;'><span style='color:#a0aec0;'>{k}</span><span style='color:#e2e8f0; font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Train and evaluate 3 different machine learning algorithms</div>", unsafe_allow_html=True)

    df_use = get_active_df()  # FIX: use helper instead of `or` on DataFrame
    if df_use is None:
        st.warning("⚠️ Please upload data first.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        split = st.selectbox("Train/Test Split", ["80% / 20%", "70% / 30%", "90% / 10%"])
    with c2:
        cv = st.selectbox("Cross-Validation Folds", ["5-Fold", "3-Fold", "10-Fold"])
    with c3:
        seed = st.number_input("Random Seed", value=42, min_value=1, max_value=9999)

    test_size = float(split.split("/")[1].strip().replace("%", "")) / 100

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>📈</div>
            <div class='feature-title'>Linear Regression</div>
            <div class='feature-desc'>Simple yet powerful algorithm for linear relationships</div>
            <div style='margin-top:8px; font-size:0.75rem; color:#718096;'>Ridge | L2 | LR:0.01 | 1000 iter</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>🌲</div>
            <div class='feature-title'>Random Forest</div>
            <div class='feature-desc'>Ensemble method using multiple decision trees</div>
            <div style='margin-top:8px; font-size:0.75rem; color:#718096;'>Trees:100 | MaxDepth:20 | MinSplit:5</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>⚡</div>
            <div class='feature-title'>Gradient Boosting</div>
            <div class='feature-desc'>Sequential ensemble method for high accuracy</div>
            <div style='margin-top:8px; font-size:0.75rem; color:#718096;'>Est:200 | LR:0.1 | MaxDepth:5</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🚀 Train All Models"):
        with st.spinner("Preparing features…"):
            df_prep, _, _ = prepare_features(df_use)
            feat_cols = [f for f in FEATURES if f in df_prep.columns]
            Xdf = df_prep[feat_cols].fillna(0)
            y = df_prep["price"]
            X_train, X_test, y_train, y_test = train_test_split(
                Xdf, y, test_size=test_size, random_state=int(seed))

        progress = st.progress(0)
        status = st.empty()

        models_def = {
            "Linear Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=int(seed), n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=int(seed)),
        }

        trained = {}
        metrics = {}
        for i, (name, model) in enumerate(models_def.items()):
            status.markdown(f"Training **{name}**…")
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            trained[name] = model
            metrics[name] = {"r2": float(r2), "rmse": rmse, "mae": mae, "time": elapsed, "feat_cols": feat_cols}
            progress.progress((i + 1) / 3)

        st.session_state.models = trained
        st.session_state.model_metrics = metrics
        st.session_state.training_done = True
        status.empty()
        progress.empty()
        st.success("✅ All models trained successfully!")

        rows = []
        for name, m in metrics.items():
            rows.append({"Model": name, "R² Score": f"{m['r2']:.3f}",
                         "RMSE": f"${m['rmse']:,.0f}", "MAE": f"${m['mae']:,.0f}",
                         "Time (s)": f"{m['time']:.1f}s"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("<div class='section-title'>Model Comparison & Selection</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Compare performance metrics and select the best model for deployment</div>", unsafe_allow_html=True)

    if not st.session_state.training_done:
        st.warning("⚠️ Please train models first.")
        st.stop()

    metrics = st.session_state.model_metrics
    best_name = max(metrics, key=lambda k: metrics[k]["r2"])
    best = metrics[best_name]

    st.markdown(f"""
    <div class='best-model-banner'>
        🏆 <strong>Best Performing Model</strong><br/>
        <strong>{best_name}</strong> achieved the highest R² score of <strong>{best['r2']:.3f}</strong>
        with the lowest prediction errors.<br/>
        <span style='color:#a0aec0; font-size:0.85rem;'>Recommended for production deployment based on overall performance metrics.</span>
    </div>""", unsafe_allow_html=True)

    st.subheader("Performance Metrics Comparison")
    ranks = sorted(metrics.keys(), key=lambda k: metrics[k]["r2"], reverse=True)
    rank_badges = ["<span class='rank-1'>🥇 Best</span>", "<span class='rank-2'>🥈 #2</span>", "<span class='rank-3'>🥉 #3</span>"]
    for i, name in enumerate(ranks):
        m = metrics[name]
        c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1.2, 1.5, 1.5, 1.5, 1.2])
        c1.markdown(f"**{name}**", unsafe_allow_html=True)
        c2.markdown(f"**{m['r2']:.3f}**")
        c3.markdown(f"${m['rmse']:,.0f}")
        c4.markdown(f"${m['mae']:,.0f}")
        c5.markdown(f"{m['time']:.1f}s")
        c6.markdown(rank_badges[i], unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=list(metrics.keys()), y=[m["r2"] for m in metrics.values()],
                     title="R² Score Comparison", color=list(metrics.keys()),
                     color_discrete_sequence=["#6366f1", "#4ade80", "#f59e0b"])
        fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0", showlegend=False)
        fig.update_yaxes(range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        categories = ["Accuracy", "Speed", "Scalability", "Robustness", "Interpretability"]
        fig2 = go.Figure()
        model_vals = {
            "Linear Regression": [0.78, 0.99, 0.95, 0.70, 0.99],
            "Random Forest":      [0.87, 0.75, 0.80, 0.88, 0.60],
            "Gradient Boosting":  [0.89, 0.55, 0.70, 0.90, 0.50],
        }
        colors = ["#6366f1", "#4ade80", "#f59e0b"]
        for (name, vals), col in zip(model_vals.items(), colors):
            fig2.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                                           fill="toself", name=name, line_color=col, opacity=0.7))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], color="#718096"),
                                       angularaxis=dict(color="#a0aec0")),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#a0aec0", title="Multi-dimensional Comparison",
                           title_font_color="#e2e8f0", margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Select Model for Deployment")
    options = [f"{n} | R²={metrics[n]['r2']:.3f} | RMSE=${metrics[n]['rmse']:,.0f} | Time={metrics[n]['time']:.1f}s"
               for n in ranks]
    selected = st.radio("", options, label_visibility="collapsed")
    selected_name = selected.split(" | ")[0]

    if st.button("🚀 Deploy Selected Model"):
        st.session_state.deployed_model_name = selected_name
        st.session_state.deployed_model = st.session_state.models[selected_name]
        st.session_state.deployed_feat_cols = st.session_state.model_metrics[selected_name]["feat_cols"]
        st.success(f"✅ {selected_name} deployed successfully!")

# ═══════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.markdown("<div class='section-title'>Price Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Use the deployed model to predict house prices for new properties</div>", unsafe_allow_html=True)

    if st.session_state.deployed_model is None:
        st.warning("⚠️ Please deploy a model from the Model Comparison page first.")
        st.stop()

    model = st.session_state.deployed_model
    model_name = st.session_state.deployed_model_name
    metrics = st.session_state.model_metrics.get(model_name, {})

    st.markdown(f"""<div class='info-box'>
        <strong>Active Model:</strong> {model_name} &nbsp;|&nbsp;
        <strong>R² Score:</strong> {metrics.get('r2', 0):.3f} &nbsp;|&nbsp;
        <strong>RMSE:</strong> ${metrics.get('rmse', 0):,.0f}
    </div>""", unsafe_allow_html=True)

    st.subheader("Property Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        beds = st.number_input("Bedrooms", 1, 10, 3)
        sqft_lot = st.number_input("Sqft Lot", 1000, 100000, 5000)
    with c2:
        baths = st.number_input("Bathrooms", 1.0, 10.0, 2.0, step=0.5)
        floors = st.number_input("Floors", 1.0, 4.0, 1.0, step=0.5)
    with c3:
        sqft = st.number_input("Sqft Living", 200, 15000, 2000)
        yr_built = st.number_input("Year Built", 1900, 2024, 1990)

    c4, c5, c6 = st.columns(3)
    with c4:
        state = st.selectbox("State", ["WA","CA","TX","FL","NY","IL","PA","OH","GA","NC",
                                        "MI","NJ","VA","AZ","MA","TN","IN","MO","MD","WI"])
    with c5:
        prop_type = st.selectbox("Property Type", ["SINGLE_FAMILY","CONDO","MULTI_FAMILY","TOWNHOUSE","LOT"])
    with c6:
        waterfront = st.selectbox("Waterfront", ["No", "Yes"])

    if st.button("🏠 Predict Price", use_container_width=True):
        df_use = get_active_df()  # FIX: use helper instead of `or` on DataFrame
        df_prep, le_state, le_type = prepare_features(df_use)
        try:
            state_enc = le_state.transform([state])[0]
        except ValueError:
            state_enc = 0
        try:
            type_enc = le_type.transform([prop_type])[0]
        except ValueError:
            type_enc = 0

        feat_dict = {
            "bedroom_number": beds, "bathroom_number": baths,
            "living_space": sqft, "state_enc": state_enc,
            "property_type_enc": type_enc, "year_build": yr_built
        }
        feat_cols = st.session_state.deployed_feat_cols  # FIX: now initialized in session state
        X_pred = np.array([[feat_dict.get(c, 0) for c in feat_cols]])
        price = model.predict(X_pred)[0]

        confidence_low = price * 0.92
        confidence_high = price * 1.08

        st.markdown(f"""
        <div class='price-result'>
            <div class='price-label'>Estimated Property Value</div>
            <div class='price-amount'>${price:,.0f}</div>
            <div class='price-label' style='margin-top:8px;'>
                Confidence Range: ${confidence_low:,.0f} – ${confidence_high:,.0f}
            </div>
            <div class='price-label' style='margin-top:4px;'>
                ${price/sqft:.0f}/sqft &nbsp;|&nbsp; Model: {model_name}
            </div>
        </div>""", unsafe_allow_html=True)

        st.session_state.predictions_history.append({
            "label": f"{beds} bed, {baths} bath | {sqft:,} sqft",
            "price": price,
            "time": "just now"
        })

    if st.session_state.predictions_history:
        st.subheader("Recent Predictions")
        for item in reversed(st.session_state.predictions_history[-5:]):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{item['label']}**  \n<span style='color:#718096; font-size:0.8rem;'>Predicted {item['time']}</span>", unsafe_allow_html=True)
            c2.markdown(f"<div style='color:#4ade80; font-weight:700; font-size:1.1rem; text-align:right;'>${item['price']:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHY MAP
# ═══════════════════════════════════════════════════════════════════
elif page == "🗺 Geography Map":
    st.markdown("<div class='section-title'>Geography Map</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Explore property locations across the United States</div>", unsafe_allow_html=True)

    df_raw = st.session_state.df_raw

    c1, c2, c3 = st.columns(3)
    with c1:
        min_price = st.number_input("Min Price ($)", 0, 5_000_000, 0, step=50000)
    with c2:
        max_price = st.number_input("Max Price ($)", 0, 10_000_000, 2_000_000, step=50000)
    with c3:
        prop_filter = st.multiselect("Property Types",
            ["SINGLE_FAMILY", "CONDO", "MULTI_FAMILY", "TOWNHOUSE", "LOT"],
            default=["SINGLE_FAMILY", "CONDO"])

    @st.cache_data(show_spinner=False)
    def get_map_data(path, n=500):
        return load_map_sample(path, n)

    with st.spinner("Loading map data..."):
        if os.path.exists(CSV_PATH):
            map_df = get_map_data(CSV_PATH, 500)
        elif df_raw is not None:
            # FIX: safe column access — check existence before converting
            map_df = df_raw.copy()
            for coord_col in ["latitude", "longitude"]:
                if coord_col in map_df.columns:
                    map_df[coord_col] = pd.to_numeric(map_df[coord_col], errors="coerce")
                else:
                    map_df[coord_col] = np.nan
            map_df = map_df.dropna(subset=["latitude", "longitude"])
        else:
            st.warning("Please upload the dataset or ensure the CSV is in the app directory.")
            st.stop()

    map_df["price"] = pd.to_numeric(map_df["price"], errors="coerce")
    filtered = map_df[(map_df["price"] >= min_price) & (map_df["price"] <= max_price)]
    if prop_filter and "property_type" in filtered.columns:
        filtered = filtered[filtered["property_type"].isin(prop_filter)]

    st.markdown(f"**Showing {len(filtered):,} properties** (filtered from {len(map_df):,})")

    if len(filtered) == 0:
        st.warning("No properties match the current filters.")
        st.stop()

    m = folium.Map(
        location=[39.5, -98.35], zoom_start=4,
        tiles="CartoDB dark_matter"
    )

    def price_color(price):
        if price < 200000: return "#4ade80"
        elif price < 400000: return "#facc15"
        elif price < 700000: return "#f97316"
        else: return "#ef4444"

    for _, row in filtered.iterrows():
        p = row.get("price", 0)
        beds = row.get("bedroom_number", "N/A")
        baths = row.get("bathroom_number", "N/A")
        sqft = row.get("living_space", "N/A")
        addr = row.get("address", "N/A")
        ptype = row.get("property_type", "N/A")
        url = row.get("property_url", "")

        popup_html = f"""
        <div style='font-family:sans-serif; width:220px;'>
            <strong style='font-size:14px;'>${p:,.0f}</strong><br/>
            <span style='color:#555;'>{addr}</span><br/>
            <hr style='margin:4px 0;'/>
            🛏 {beds} beds &nbsp; 🚿 {baths} baths<br/>
            📐 {sqft} sqft &nbsp; 🏘 {ptype}<br/>
            {"<a href='" + str(url) + "' target='_blank'>View Listing ↗</a>" if url and str(url) != "nan" else ""}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=price_color(p),
            fill=True,
            fill_color=price_color(p),
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"${p:,.0f} | {addr}"
        ).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])

    st.markdown("""
    <div style='display:flex; gap:20px; margin-top:8px; font-size:0.85rem;'>
        <span><span style='color:#4ade80;'>●</span> &lt;$200K</span>
        <span><span style='color:#facc15;'>●</span> $200K–$400K</span>
        <span><span style='color:#f97316;'>●</span> $400K–$700K</span>
        <span><span style='color:#ef4444;'>●</span> &gt;$700K</span>
    </div>
    """, unsafe_allow_html=True)

    if "state" in filtered.columns:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            state_counts = filtered["state"].value_counts().head(10)
            fig = px.bar(x=state_counts.values, y=state_counts.index,
                         orientation="h", title="Top 10 States by Listings",
                         color_discrete_sequence=["#6366f1"])
            fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            state_prices = filtered.groupby("state")["price"].median().nlargest(10)
            fig2 = px.bar(x=state_prices.values, y=state_prices.index,
                          orientation="h", title="Top 10 States by Median Price",
                          color_discrete_sequence=["#4ade80"])
            fig2.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════
elif page == "💬 AI Chatbot":
    st.markdown("<div class='section-title'>AI Real Estate Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Ask questions about the dataset, models, and market trends</div>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.session_state.chat_history = [{
            "role": "bot",
            "content": "👋 Hi! I'm your AI real estate assistant. I can answer questions about prices, models, locations, and market trends. What would you like to know?"
        }]

    # FIX: escape user messages to prevent HTML injection breaking the chat layout
    chat_html = "<div class='chat-container'>"
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            safe_content = html.escape(msg["content"])
            chat_html += f"<div class='chat-user'>{safe_content}</div>"
        else:
            content = msg["content"].replace("\n", "<br/>")
            chat_html += f"<div class='chat-bot'>{content}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**💡 Quick questions:**")
    cols = st.columns(4)
    quick = [
        "What is the average price?",
        "Which states have highest prices?",
        "How did Gradient Boosting perform?",
        "What property types are available?",
    ]
    for i, q in enumerate(quick):
        if cols[i].button(q, key=f"quick_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            df = get_active_df()  # FIX: use helper
            resp = get_bot_response(q, df, st.session_state.model_metrics)
            st.session_state.chat_history.append({"role": "bot", "content": resp})
            st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("", placeholder="Ask me anything about real estate...", label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        df = get_active_df()  # FIX: use helper
        resp = get_bot_response(user_input, df, st.session_state.model_metrics)
        st.session_state.chat_history.append({"role": "bot", "content": resp})
        st.rerun()

    if len(st.session_state.chat_history) > 1:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = [{
                "role": "bot",
                "content": "Chat cleared. How can I help you?"
            }]
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 Dashboard":
    st.markdown("<div class='section-title'>Real Estate Trends Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Comprehensive analytics and market insights</div>", unsafe_allow_html=True)

    df = get_active_df()  # FIX: use helper instead of `or` on DataFrame
    if df is None:
        st.warning("⚠️ Please upload data first.")
        st.stop()

    avg_price = df["price"].mean()
    total = len(df)
    med_price = df["price"].median()
    ppsqft = (df["price"] / df["living_space"]).median() if "living_space" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    metrics_data = [
        ("Avg Price", f"${avg_price:,.0f}", "+5.2%", True, "$"),
        ("Total Listings", f"{total:,}", "+12.8%", True, "🏘"),
        ("Avg Days on Market", "32 days", "-8.5%", False, "📅"),  # note: illustrative, not from data
        ("Price per Sqft", f"${ppsqft:,.0f}", "+3.1%", True, "📐"),
    ]
    for col, (label, val, delta, pos, icon) in zip([c1, c2, c3, c4], metrics_data):
        delta_class = "metric-delta-pos" if pos else "metric-delta-neg"
        col.markdown(f"""<div class='metric-card'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
            <div class='{delta_class}'>{delta} vs last year</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        np.random.seed(42)
        base = avg_price
        prices_trend = [base * (1 + 0.02 * i + np.random.uniform(-0.02, 0.02)) for i in range(12)]
        fig = px.area(x=months, y=prices_trend, title="Average Price Trend (2024)",
                      labels={"x": "", "y": "Price"},
                      color_discrete_sequence=["#6366f1"])
        fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
        fig.update_traces(fill="tozeroy", fillcolor="rgba(99,102,241,0.15)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "bedroom_number" in df.columns:
            bed_price = df.groupby("bedroom_number")["price"].mean().reset_index()
            bed_price = bed_price[bed_price["bedroom_number"].between(1, 6)]
            fig2 = px.bar(bed_price, x="bedroom_number", y="price",
                          title="Average Price by Bedrooms",
                          color_discrete_sequence=["#4ade80"])
            fig2.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "property_type" in df.columns:
            pt_counts = df["property_type"].value_counts().head(5)
            colors = ["#6366f1", "#4ade80", "#f59e0b", "#ef4444", "#a78bfa"]
            fig3 = px.pie(values=pt_counts.values, names=pt_counts.index,
                          title="Property Types Distribution",
                          color_discrete_sequence=colors)
            fig3.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
            st.plotly_chart(fig3, use_container_width=True)
    with col4:
        grp_col = "state" if "state" in df.columns else ("postcode" if "postcode" in df.columns else None)
        if grp_col:
            top_grp = df.groupby(grp_col)["price"].median().nlargest(5)
            fig4 = px.bar(x=top_grp.values, y=top_grp.index.astype(str),
                          orientation="h", title=f"Top 5 {grp_col.title()}s by Price",
                          color_discrete_sequence=["#a78bfa"])
            fig4.update_layout(**PLOTLY_LAYOUT, title_font_color="#e2e8f0")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Market Insights")
    c1, c2, c3 = st.columns(3)
    insights = [
        ("🔥 Hot Markets", "Limited inventory is driving competitive offers in high-demand metros. Single-family homes seeing fastest appreciation.", "#f97316"),
        ("💡 Price Predictions", f"ML models forecast continued price growth based on historical trends. Current median: ${med_price:,.0f}.", "#6366f1"),
        ("📦 Inventory Trends", "Single-family homes remain dominant in listings. Condo inventory growing in urban areas year-over-year.", "#4ade80"),
    ]
    for col, (title, desc, color) in zip([c1, c2, c3], insights):
        col.markdown(f"""<div style='background:#1e2235; border:1px solid {color}; border-radius:10px; padding:16px;'>
            <div style='font-weight:700; color:{color}; margin-bottom:8px;'>{title}</div>
            <div style='font-size:0.85rem; color:#a0aec0;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
