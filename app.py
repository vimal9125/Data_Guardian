# guardian_ai_v2.py — FIXED & ENHANCED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Sidebar / Theme
# ------------------------------
# Sidebar / Theme (Enhanced)
# ------------------------------
# ------------------------------
# Sidebar / Theme (Enhanced + Fixed Logo)
# ------------------------------
with st.sidebar:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #111827;
                padding: 1.8rem 1rem;
                border-right: 1px solid #1f2937;
            }
            .sidebar-logo {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 0.8rem;
            }
            .sidebar-header {
                font-size: 1.8rem;
                font-weight: 800;
                color: #f9fafb;
                text-align: center;
                margin-bottom: 0.3rem;
            }
            .sidebar-sub {
                font-size: 1rem;
                color: #9ca3af;
                text-align: center;
                margin-bottom: 1.5rem;
            }
            .sidebar-section {
                font-size: 1.1rem;
                font-weight: 600;
                color: #f3f4f6;
                margin-top: 1.3rem;
                border-bottom: 1px solid #374151;
                padding-bottom: 0.4rem;
            }
            .sidebar-tip {
                font-size: 0.9rem;
                color: #9ca3af;
                margin-top: 1.5rem;
                background-color: #1f2937;
                padding: 0.8rem;
                border-radius: 0.6rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Stable, always-visible logo (can replace with your own)
    st.markdown(
        "<div class='sidebar-logo'><img src='https://i.ibb.co/4gVJj1C/data-guardian-logo.png' width='100'></div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='sidebar-header'>🧠 DataGuardian AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Automated Data Auditing & Quality Insights</div>", unsafe_allow_html=True)

    theme_dark = st.toggle("🌙 Dark Mode", value=True)

    st.markdown("<div class='sidebar-section'>⚙️ Controls</div>", unsafe_allow_html=True)
    contamination = st.slider("Anomaly Contamination (IsolationForest)", 0.01, 0.2, 0.05, 0.01)

    st.markdown("<div class='sidebar-tip'>💡 Tip: Upload CSV/XLSX files. Large datasets may take longer to process.</div>", unsafe_allow_html=True)


# ------------------------------
# Theme CSS
# ------------------------------
if theme_dark:
    base_bg = "#0e1117"
    card_bg = "#111318"
    text_color = "#e6eef8"
    accent = "#00e676"
else:
    base_bg = "#f7f9fb"
    card_bg = "#ffffff"
    text_color = "#0b1724"
    accent = "#0072ff"

# ✅ FIX: CSS block wrapped inside triple quotes
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {base_bg};
            color: {text_color};
        }}
        .card {{
            background-color: {card_bg};
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        }}
        .metric-card {{
            padding: 14px;
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(0,0,0,0.05), rgba(0,0,0,0.1));
        }}
        .big-title {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
            color: {text_color};
        }}
        .subtle {{
            color: rgba(200,200,200,0.7);
            font-size: 13px;
        }}
        /* ✅ FIX: Wrap CSS correctly */
        .stDataFrame > div {{
            overflow-x: auto;
        }}
        .stProgress > div > div > div {{
            background-color: {accent} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Top header
# ------------------------------
colH1, colH2 = st.columns([6, 1])
with colH1:
    st.markdown(f"<div class='big-title'>🧠 DataGuardian AI — Enhanced Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Professional analytics view · Clean visuals · Actionable insights</div>", unsafe_allow_html=True)
with colH2:
    st.markdown(f"<div style='text-align:right; color:{text_color}; font-size:12px;'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# File upload
# ------------------------------
uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded is None:
    st.info("📁 Please upload a CSV or XLSX file to begin the audit.")
    st.stop()

# ------------------------------
# Read file
# ------------------------------
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

# Basic cleanup
df_original = df.copy()
n_rows, n_cols = df.shape
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📉 Missing Values", "🚨 Anomalies", "🧠 AI Summary"])

# ------------------------------
# Overview Tab
# ------------------------------
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{n_rows:,}")
    c2.metric("Total Columns", f"{n_cols:,}")
    c3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    c4.metric("Columns with ≤1 unique value", f"{sum(df.nunique() < 2)}")

    st.markdown("### 📊 Sample (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    if len(numeric_cols) > 1:
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            origin="lower",
            title="Correlation matrix"
        )
        fig_corr.update_layout(height=500, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------
# Missing Values Tab
# ------------------------------
with tab2:
    st.subheader("Missing Value Analysis")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        st.success("🎉 No missing values found!")
    else:
        miss_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / n_rows * 100).round(2)
        })
        fig_miss = px.bar(
            miss_df,
            x="Missing %",
            y="Column",
            orientation="h",
            title="Missing Values (%) by Column",
            color="Missing %",
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_miss, use_container_width=True)
        st.dataframe(miss_df, use_container_width=True)

# ------------------------------
# Anomalies Tab
# ------------------------------
with tab3:
    st.subheader("Anomaly Detection (Isolation Forest)")
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for anomaly detection.")
    else:
        df_proc = df[numeric_cols].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_proc)

        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(scaled)
        df["Anomaly"] = np.where(preds == -1, "Anomaly", "Normal")

        anom_count = (df["Anomaly"] == "Anomaly").sum()
        normal_count = (df["Anomaly"] == "Normal").sum()

        colm1, colm2, colm3 = st.columns([1,1,2])
        colm1.metric("Total Anomalies", anom_count)
        colm2.metric("Normal Records", normal_count)

        fig_pie = px.pie(
            names=["Normal", "Anomaly"],
            values=[normal_count, anom_count],
            color=["Normal", "Anomaly"],
            color_discrete_map={"Normal": "green", "Anomaly": "red"},
            title="Normal vs Anomaly Distribution",
            hole=0.4
        )
        colm3.plotly_chart(fig_pie, use_container_width=True)

        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(scaled)
        pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        pc_df["Anomaly"] = df["Anomaly"]

        fig_scatter = px.scatter(
            pc_df,
            x="PC1",
            y="PC2",
            color="Anomaly",
            symbol="Anomaly",
            title="PCA 2D: Anomaly vs Normal",
            color_discrete_map={"Anomaly": "red", "Normal": "green"}
        )
        fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# AI Summary Tab
# ------------------------------
with tab4:
    st.subheader("AI Summary & Recommendations")

    total_cells = n_rows * n_cols
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = (missing_cells / total_cells * 100)
    health = 100 - min(40, missing_pct)
    anomalies_pct = (df["Anomaly"] == "Anomaly").mean() * 100 if "Anomaly" in df else 0
    health -= anomalies_pct

    st.metric("Dataset Health", f"{int(health)} / 100")
    st.progress(int(health))
    st.markdown(f"📊 Missing values: **{missing_pct:.2f}%**, anomalies: **{anomalies_pct:.2f}%**.")

    st.write("""
    **Recommended Actions:**
    - Handle missing values via imputation or dropping high-missing columns.
    - Investigate anomaly clusters for data drift.
    - Remove features with no variance.
    - Use PCA or feature selection for correlated variables.
    """)

st.caption("Built with ❤️ — DataGuardian AI v2.0 (Enhanced)")
