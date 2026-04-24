import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent App Testing System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid #2d2d5e;
}
[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}

/* Main bg */
.stApp {
    background: #0b0b14;
    color: #e0e0ff;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #7c83fd !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #9090cc !important; }

/* Headers */
h1, h2, h3 { color: #c0c0ff !important; font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #2d2d5e;
    border-radius: 8px;
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #1e1e3f 100%);
    border: 1px solid #3d3d7e;
    border-left: 4px solid #7c83fd;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.92rem;
    color: #d0d0ff;
}

.risk-high {
    border-left-color: #ff4d6d !important;
    background: linear-gradient(135deg, #2a1a1e 0%, #1e1a2e 100%) !important;
}

.risk-medium {
    border-left-color: #ffd166 !important;
    background: linear-gradient(135deg, #2a2a1a 0%, #1e1e2e 100%) !important;
}

.risk-low {
    border-left-color: #06d6a0 !important;
    background: linear-gradient(135deg, #1a2a1e 0%, #1a1e2e 100%) !important;
}

/* Priority badge */
.priority-badge {
    display: inline-block;
    background: #7c83fd22;
    color: #7c83fd;
    border: 1px solid #7c83fd;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 8px;
}

/* Divider */
hr { border-color: #2d2d5e !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #9090cc;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #7c83fd22 !important;
    color: #c0c0ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c83fd, #5c63cd);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9ca3fe, #7c83fd);
}

/* Selectbox, slider */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #1a1a2e;
    border: 1px solid #2d2d5e;
    color: #e0e0ff;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_COLORS = ["#7c83fd", "#ff4d6d", "#ffd166", "#06d6a0", "#a29bfe",
                 "#fd79a8", "#55efc4", "#fdcb6e", "#74b9ff", "#e17055"]


def plotly_layout(fig, title="", height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(15,15,26,0)",
        plot_bgcolor="rgba(15,15,26,0)",
        font=dict(family="Syne, sans-serif", color="#c0c0ff"),
        title=dict(text=title, font=dict(size=16, color="#c0c0ff")),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        try:
            df = pd.read_csv("bugs_data.csv")
        except FileNotFoundError:
            st.error("bugs_data.csv not found. Please upload a file or run generate_data.py.")
            st.stop()
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df["Report Date"] = pd.to_datetime(df["Report Date"])
    return df


@st.cache_data
def compute_risk_scores(df):
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    df2 = df.copy()
    df2["SeverityWeight"] = df2["Severity"].map(severity_map)

    grp = df2.groupby("Module").agg(
        TotalBugs=("Bug ID", "count"),
        TotalOccurrences=("Occurrences", "sum"),
        AvgSeverity=("SeverityWeight", "mean"),
        ReopenedCount=("Status", lambda x: (x == "Reopened").sum()),
        AvgFixTime=("Time to Fix (days)", "mean"),
    ).reset_index()

    grp["ReopenRate"] = grp["ReopenedCount"] / grp["TotalBugs"]

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    grp["RiskScore"] = (
        norm(grp["TotalOccurrences"]) * 0.35
        + norm(grp["AvgSeverity"]) * 0.35
        + norm(grp["ReopenRate"]) * 0.30
    ) * 100

    grp["RiskScore"] = grp["RiskScore"].round(1)
    grp["RiskLevel"] = pd.cut(
        grp["RiskScore"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )
    return grp.sort_values("RiskScore", ascending=False).reset_index(drop=True)


@st.cache_data
def train_model(df):
    df2 = df.copy()
    le_mod = LabelEncoder()
    le_sev = LabelEncoder()
    le_stat = LabelEncoder()
    le_ver = LabelEncoder()

    df2["ModuleEnc"] = le_mod.fit_transform(df2["Module"])
    df2["SeverityEnc"] = le_sev.fit_transform(df2["Severity"])
    df2["VersionEnc"] = le_ver.fit_transform(df2["App Version"])
    df2["StatusEnc"] = le_stat.fit_transform(df2["Status"])

    target_col = "StatusEnc"
    feature_cols = ["ModuleEnc", "SeverityEnc", "VersionEnc",
                    "Occurrences", "Time to Fix (days)"]

    X = df2[feature_cols]
    y = df2[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                       learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=le_stat.classes_,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(
        model.feature_importances_,
        index=["Module", "Severity", "Version", "Occurrences", "Time to Fix"]
    ).sort_values(ascending=True)

    encoders = {
        "module": le_mod,
        "severity": le_sev,
        "version": le_ver,
        "status": le_stat,
    }
    return model, acc, report, cm, importances, feature_cols, encoders


@st.cache_data
def run_nlp(df):
    descriptions = df["Bug Description"].dropna().tolist()
    if len(descriptions) < 5:
        return None, None, None

    tfidf = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(descriptions)

    n_clusters = min(6, len(descriptions) // 10)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_tfidf)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_tfidf.toarray())

    feature_names = tfidf.get_feature_names_out()
    tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).flatten()
    top_terms = pd.Series(tfidf_sum, index=feature_names).sort_values(ascending=False).head(25)

    cluster_df = pd.DataFrame({
        "Bug Description": descriptions,
        "Cluster": labels,
        "X": coords[:, 0],
        "Y": coords[:, 1],
    })
    return top_terms, cluster_df, n_clusters


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
  <div style='font-size:2.5rem'>🔬</div>
  <div style='font-size:1.1rem; font-weight:800; color:#c0c0ff; letter-spacing:1px;'>
    IATS
  </div>
  <div style='font-size:0.72rem; color:#6060aa; letter-spacing:2px; margin-top:2px;'>
    INTELLIGENT APP TESTING
  </div>
</div>
<hr style='border-color:#2d2d5e; margin:8px 0 20px 0;'/>
""", unsafe_allow_html=True)

pages = {
    "📋 Data Overview": "data_overview",
    "📊 Exploratory Analysis": "eda",
    "⚠️ Risk Scoring": "risk",
    "🤖 Bug Prediction Model": "model",
    "🔤 NLP Bug Analysis": "nlp",
    "✅ Fix Validation & Insights": "insights",
}
page_label = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
page = pages[page_label]

st.sidebar.markdown("<hr style='border-color:#2d2d5e; margin:20px 0;'/>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📂 Upload bugs_data.csv", type=["csv"])
df = load_data(uploaded_file)

st.sidebar.markdown(f"""
<div style='padding:12px; background:#1a1a2e; border-radius:8px; border:1px solid #2d2d5e; margin-top:10px;'>
  <div style='font-size:0.72rem; color:#6060aa; letter-spacing:1px; margin-bottom:8px;'>DATASET INFO</div>
  <div style='font-size:0.85rem; color:#c0c0ff;'>🗃️ {len(df):,} bugs loaded</div>
  <div style='font-size:0.85rem; color:#c0c0ff;'>📦 {df['App Version'].nunique()} versions</div>
  <div style='font-size:0.85rem; color:#c0c0ff;'>🧩 {df['Module'].nunique()} modules</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "data_overview":
    st.title("📋 Data Overview")
    st.markdown("Explore the raw bug dataset, understand its structure, and review data quality.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bugs", f"{len(df):,}")
    c2.metric("App Versions", df["App Version"].nunique())
    c3.metric("Modules", df["Module"].nunique())
    c4.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("#### Raw Data")
    version_filter = st.multiselect("Filter by Version", df["App Version"].unique(),
                                    default=list(df["App Version"].unique()))
    filtered = df[df["App Version"].isin(version_filter)] if version_filter else df
    st.dataframe(filtered, use_container_width=True, height=360)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Type": df.dtypes.values.astype(str),
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Numeric Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("#### Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(12, 2))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    miss = df.isnull().astype(int)
    if miss.values.sum() == 0:
        ax.text(0.5, 0.5, "✓ No missing values in the dataset",
                ha="center", va="center", color="#06d6a0", fontsize=14,
                transform=ax.transAxes)
        ax.axis("off")
    else:
        sns.heatmap(miss, ax=ax, cbar=False, cmap="YlOrRd", yticklabels=False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "eda":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # Bug count by module
    mod_counts = df["Module"].value_counts().reset_index()
    mod_counts.columns = ["Module", "Count"]
    fig1 = px.bar(mod_counts, x="Count", y="Module", orientation="h",
                  color="Count", color_continuous_scale="Plasma",
                  labels={"Count": "Bug Count"})
    plotly_layout(fig1, "Bug Count by Module", height=380)
    fig1.update_coloraxes(showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sev_counts = df["Severity"].value_counts()
        fig2 = px.pie(values=sev_counts.values, names=sev_counts.index,
                      color=sev_counts.index,
                      color_discrete_map={"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#06d6a0"},
                      hole=0.45)
        plotly_layout(fig2, "Severity Distribution", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        stat_counts = df["Status"].value_counts()
        fig3 = px.pie(values=stat_counts.values, names=stat_counts.index,
                      color=stat_counts.index,
                      color_discrete_map={"Fixed": "#06d6a0", "Open": "#ffd166", "Reopened": "#ff4d6d"},
                      hole=0.45)
        plotly_layout(fig3, "Bug Status Breakdown", height=360)
        st.plotly_chart(fig3, use_container_width=True)

    # Bug trends across versions
    trend = df.groupby(["App Version", "Severity"]).size().reset_index(name="Count")
    fig4 = px.line(trend, x="App Version", y="Count", color="Severity",
                   markers=True,
                   color_discrete_map={"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#06d6a0"})
    plotly_layout(fig4, "Bug Trends Across Versions by Severity", height=380)
    st.plotly_chart(fig4, use_container_width=True)

    # Top 10 most occurring bugs
    st.markdown("#### Top 10 Most Frequently Occurring Bugs")
    top10 = df.nlargest(10, "Occurrences")[
        ["Bug ID", "Module", "Bug Description", "Severity", "Occurrences", "Status"]
    ].reset_index(drop=True)
    st.dataframe(top10, use_container_width=True, hide_index=True)

    fig5 = px.bar(top10, x="Occurrences", y="Bug ID", orientation="h",
                  color="Severity",
                  color_discrete_map={"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#06d6a0"},
                  hover_data=["Module", "Bug Description"])
    plotly_layout(fig5, "Top 10 Bugs by Occurrence Count", height=380)
    st.plotly_chart(fig5, use_container_width=True)

    # Occurrences vs Fix Time scatter
    fig6 = px.scatter(df, x="Occurrences", y="Time to Fix (days)",
                      color="Severity", size="Occurrences",
                      facet_col="Status", hover_data=["Module", "Bug ID"],
                      color_discrete_map={"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#06d6a0"})
    plotly_layout(fig6, "Occurrences vs Time to Fix (by Status)", height=420)
    st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "risk":
    st.title("⚠️ Risk Scoring System")
    st.markdown("Risk score = **35% Occurrence Frequency** + **35% Severity Weight** + **30% Reopen Rate**")
    st.markdown("---")

    risk_df = compute_risk_scores(df)

    c1, c2, c3 = st.columns(3)
    high_risk = (risk_df["RiskLevel"] == "High").sum()
    med_risk = (risk_df["RiskLevel"] == "Medium").sum()
    low_risk = (risk_df["RiskLevel"] == "Low").sum()
    c1.metric("🔴 High Risk Modules", high_risk)
    c2.metric("🟡 Medium Risk Modules", med_risk)
    c3.metric("🟢 Low Risk Modules", low_risk)

    st.markdown("#### Module Risk Score Table")

    def color_risk(val):
        colors = {"High": "color: #ff4d6d; font-weight: bold;",
                  "Medium": "color: #ffd166; font-weight: bold;",
                  "Low": "color: #06d6a0; font-weight: bold;"}
        return colors.get(val, "")

    display_cols = ["Module", "RiskScore", "RiskLevel", "TotalBugs",
                    "TotalOccurrences", "ReopenRate", "AvgFixTime"]
    display_df = risk_df[display_cols].rename(columns={
        "RiskScore": "Risk Score",
        "RiskLevel": "Risk Level",
        "TotalBugs": "Total Bugs",
        "TotalOccurrences": "Total Occurrences",
        "ReopenRate": "Reopen Rate",
        "AvgFixTime": "Avg Fix (days)",
    })
    display_df["Reopen Rate"] = display_df["Reopen Rate"].apply(lambda x: f"{x:.1%}")
    display_df["Avg Fix (days)"] = display_df["Avg Fix (days)"].round(1)

    styled = display_df.style.applymap(color_risk, subset=["Risk Level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Color-coded bar chart
    color_map = {"High": "#ff4d6d", "Medium": "#ffd166", "Low": "#06d6a0"}
    bar_colors = [color_map[lv] for lv in risk_df["RiskLevel"]]

    fig = go.Figure(go.Bar(
        x=risk_df["Module"],
        y=risk_df["RiskScore"],
        marker_color=bar_colors,
        text=risk_df["RiskScore"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y}<extra></extra>",
    ))
    plotly_layout(fig, "Module Risk Scores", height=420)
    fig.add_hline(y=66, line_dash="dash", line_color="#ff4d6d", annotation_text="High Risk Threshold")
    fig.add_hline(y=33, line_dash="dash", line_color="#ffd166", annotation_text="Medium Risk Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart for top 5 modules
    st.markdown("#### Risk Factor Radar — Top 5 Modules")
    top5 = risk_df.head(5)

    def norm_col(col):
        mn, mx = risk_df[col].min(), risk_df[col].max()
        return (top5[col] - mn) / (mx - mn + 1e-9)

    categories = ["Occurrences", "Severity", "Reopen Rate", "Fix Time", "Total Bugs"]
    radar_fig = go.Figure()
    for _, row in top5.iterrows():
        vals = [
            norm_col("TotalOccurrences").loc[row.name],
            norm_col("AvgSeverity").loc[row.name],
            norm_col("ReopenRate").loc[row.name],
            norm_col("AvgFixTime").loc[row.name],
            norm_col("TotalBugs").loc[row.name],
        ]
        vals += [vals[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["Module"],
            opacity=0.65,
        ))
    plotly_layout(radar_fig, "", height=450)
    st.plotly_chart(radar_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BUG PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "model":
    st.title("🤖 Bug Prediction Model")
    st.markdown("Gradient Boosting model trained to predict **Bug Status** (Open / Fixed / Reopened).")
    st.markdown("---")

    with st.spinner("Training model…"):
        model, acc, report, cm, importances, feature_cols, encoders = train_model(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{acc:.1%}")
    c2.metric("Training Features", len(feature_cols))
    c3.metric("Target Classes", 3)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "🔥 Confusion Matrix", "📊 Feature Importance", "🎯 Live Prediction"])

    with tab1:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        labels = encoders["status"].classes_
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, linecolor="#2d2d5e",
                    annot_kws={"size": 12, "color": "white"})
        ax.set_xlabel("Predicted", color="#c0c0ff", fontsize=11)
        ax.set_ylabel("Actual", color="#c0c0ff", fontsize=11)
        ax.tick_params(colors="#c0c0ff")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        fig_imp = px.bar(
            importances.reset_index(),
            x=importances.values,
            y=importances.index,
            orientation="h",
            color=importances.values,
            color_continuous_scale="Viridis",
        )
        plotly_layout(fig_imp, "Feature Importance", height=360)
        fig_imp.update_coloraxes(showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab4:
        st.markdown("#### Predict Bug Status")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            sel_module = st.selectbox("Module", sorted(df["Module"].unique()))
            sel_severity = st.selectbox("Severity", ["Low", "Medium", "High"])
        with pc2:
            sel_version = st.selectbox("App Version", sorted(df["App Version"].unique()))
            sel_occurrences = st.slider("Occurrences", 1, 50, 5)
        with pc3:
            sel_fix_time = st.slider("Time to Fix (days)", 0.5, 60.0, 7.0, 0.5)

        if st.button("🔮 Predict Status"):
            try:
                mod_enc = encoders["module"].transform([sel_module])[0]
                sev_enc = encoders["severity"].transform([sel_severity])[0]
                ver_enc = encoders["version"].transform([sel_version])[0]
                X_input = np.array([[mod_enc, sev_enc, ver_enc,
                                     sel_occurrences, sel_fix_time]])
                pred = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0]
                pred_label = encoders["status"].inverse_transform([pred])[0]

                color = {"Fixed": "#06d6a0", "Open": "#ffd166", "Reopened": "#ff4d6d"}.get(pred_label, "#7c83fd")
                st.markdown(f"""
                <div class='insight-card' style='border-left-color:{color}; text-align:center; padding:24px;'>
                  <div style='font-size:2rem; margin-bottom:8px;'>{'✅' if pred_label=='Fixed' else '⚠️' if pred_label=='Open' else '🔁'}</div>
                  <div style='font-size:1.6rem; font-weight:800; color:{color};'>Predicted: {pred_label}</div>
                </div>
                """, unsafe_allow_html=True)

                proba_df = pd.DataFrame({
                    "Status": encoders["status"].classes_,
                    "Probability": proba,
                })
                fig_p = px.bar(proba_df, x="Status", y="Probability",
                               color="Status",
                               color_discrete_map={"Fixed": "#06d6a0", "Open": "#ffd166", "Reopened": "#ff4d6d"})
                plotly_layout(fig_p, "Prediction Probabilities", height=300)
                fig_p.update_layout(showlegend=False)
                st.plotly_chart(fig_p, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — NLP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "nlp":
    st.title("🔤 NLP Bug Analysis")
    st.markdown("TF-IDF keyword extraction and KMeans clustering of bug descriptions.")
    st.markdown("---")

    with st.spinner("Running NLP analysis…"):
        top_terms, cluster_df, n_clusters = run_nlp(df)

    if top_terms is None:
        st.warning("Not enough data for NLP analysis.")
    else:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("#### Top TF-IDF Keywords")
            fig_kw = px.bar(
                top_terms.reset_index(),
                x=top_terms.values[::-1],
                y=top_terms.index[::-1],
                orientation="h",
                color=top_terms.values[::-1],
                color_continuous_scale="Plasma",
            )
            plotly_layout(fig_kw, "Most Common Bug Terms", height=560)
            fig_kw.update_coloraxes(showscale=False)
            st.plotly_chart(fig_kw, use_container_width=True)

        with col2:
            st.markdown("#### Bug Description Clusters (PCA 2D)")
            cluster_df["Cluster"] = cluster_df["Cluster"].astype(str)
            fig_cl = px.scatter(
                cluster_df, x="X", y="Y", color="Cluster",
                hover_data=["Bug Description"],
                color_discrete_sequence=ACCENT_COLORS,
                opacity=0.8,
            )
            plotly_layout(fig_cl, f"KMeans Clusters (k={n_clusters})", height=560)
            fig_cl.update_traces(marker=dict(size=8))
            st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown("#### Representative Bugs per Cluster")
        for i in range(n_clusters):
            cluster_bugs = cluster_df[cluster_df["Cluster"] == str(i)]["Bug Description"].head(3).tolist()
            bugs_html = "".join([f"<li>{b}</li>" for b in cluster_bugs])
            st.markdown(f"""
            <div class='insight-card'>
              <span class='priority-badge'>Cluster {i}</span>
              <strong style='color:#c0c0ff;'>Representative Bugs:</strong>
              <ul style='margin:8px 0 0 0; color:#a0a0cc;'>{bugs_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # WordCloud
        st.markdown("#### Word Frequency Treemap")
        wc_data = top_terms.reset_index()
        wc_data.columns = ["Term", "Score"]
        fig_tree = px.treemap(wc_data, path=["Term"], values="Score",
                              color="Score", color_continuous_scale="Magma")
        plotly_layout(fig_tree, "TF-IDF Term Treemap", height=400)
        st.plotly_chart(fig_tree, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FIX VALIDATION & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "insights":
    st.title("✅ Fix Validation & Recommendations")
    st.markdown("---")

    risk_df = compute_risk_scores(df)

    # ── Bugs likely to reappear ──────────────────────────────────────────────
    st.markdown("#### 🔁 Bugs Likely to Reappear")
    reopen_threshold_occ = df["Occurrences"].quantile(0.70)
    risky_bugs = df[
        (df["Status"].isin(["Reopened", "Open"])) &
        (df["Occurrences"] >= reopen_threshold_occ)
    ].sort_values("Occurrences", ascending=False).head(15)

    if risky_bugs.empty:
        st.info("No high-risk reappearing bugs found.")
    else:
        st.dataframe(
            risky_bugs[["Bug ID", "Module", "Bug Description",
                        "Severity", "Status", "Occurrences", "App Version"]],
            use_container_width=True, hide_index=True
        )

    st.markdown("---")

    # ── Version quality comparison ───────────────────────────────────────────
    st.markdown("#### 📦 Version Quality Comparison")
    ver_quality = df.groupby("App Version").agg(
        TotalBugs=("Bug ID", "count"),
        HighSeverity=("Severity", lambda x: (x == "High").sum()),
        ReopenedBugs=("Status", lambda x: (x == "Reopened").sum()),
        AvgFixTime=("Time to Fix (days)", "mean"),
        OpenBugs=("Status", lambda x: (x == "Open").sum()),
    ).reset_index()
    ver_quality["QualityScore"] = (
        100
        - ver_quality["HighSeverity"] / ver_quality["TotalBugs"] * 40
        - ver_quality["ReopenedBugs"] / ver_quality["TotalBugs"] * 30
        - (ver_quality["AvgFixTime"] / ver_quality["AvgFixTime"].max()) * 30
    ).round(1)

    fig_ver = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total Bugs", "Avg Fix Time (days)", "Quality Score"],
    )
    fig_ver.add_trace(
        go.Bar(x=ver_quality["App Version"], y=ver_quality["TotalBugs"],
               marker_color="#7c83fd", name="Total Bugs"), row=1, col=1
    )
    fig_ver.add_trace(
        go.Bar(x=ver_quality["App Version"], y=ver_quality["AvgFixTime"].round(1),
               marker_color="#ffd166", name="Avg Fix Time"), row=1, col=2
    )
    fig_ver.add_trace(
        go.Bar(x=ver_quality["App Version"], y=ver_quality["QualityScore"],
               marker_color="#06d6a0", name="Quality Score"), row=1, col=3
    )
    plotly_layout(fig_ver, "Version Quality Comparison", height=380)
    st.plotly_chart(fig_ver, use_container_width=True)

    st.dataframe(
        ver_quality.rename(columns={
            "TotalBugs": "Total Bugs", "HighSeverity": "High Sev.",
            "ReopenedBugs": "Reopened", "AvgFixTime": "Avg Fix (days)",
            "OpenBugs": "Open", "QualityScore": "Quality Score",
        }).round(1),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── Auto-generated insights ──────────────────────────────────────────────
    st.markdown("#### 💡 Actionable Insights")

    highest_risk_mod = risk_df.iloc[0]["Module"]
    highest_risk_score = risk_df.iloc[0]["RiskScore"]
    worst_version = ver_quality.sort_values("TotalBugs", ascending=False).iloc[0]["App Version"]
    fastest_fix_version = ver_quality.sort_values("AvgFixTime").iloc[0]["App Version"]
    most_reopen_mod = risk_df.sort_values("ReopenRate", ascending=False).iloc[0]["Module"]
    reopen_rate_val = risk_df.sort_values("ReopenRate", ascending=False).iloc[0]["ReopenRate"]
    highest_occ_bug = df.sort_values("Occurrences", ascending=False).iloc[0]
    total_open = (df["Status"] == "Open").sum()
    pct_high = (df["Severity"] == "High").mean() * 100
    best_quality_ver = ver_quality.sort_values("QualityScore", ascending=False).iloc[0]["App Version"]

    insights = [
        ("🔴 High Risk", f"<b>{highest_risk_mod}</b> has the highest risk score of <b>{highest_risk_score}</b>. Allocate 40%+ of QA effort here before the next release.", "risk-high"),
        ("🔁 Reopen Hotspot", f"<b>{most_reopen_mod}</b> has a reopen rate of <b>{reopen_rate_val:.0%}</b>. Root cause analysis and regression tests are strongly recommended.", "risk-high"),
        ("📦 Worst Version", f"Version <b>{worst_version}</b> had the most bugs. Review changes introduced in this release and add targeted regression coverage.", "risk-medium"),
        ("⚡ Fastest Fixes", f"Version <b>{fastest_fix_version}</b> achieved the fastest average fix time — study its sprint processes and replicate.", "risk-low"),
        ("🐛 Top Bug", f"Bug <b>{highest_occ_bug['Bug ID']}</b> in <b>{highest_occ_bug['Module']}</b> occurred <b>{highest_occ_bug['Occurrences']}</b> times. This is a critical stability issue.", "risk-high"),
        ("📂 Open Backlog", f"There are <b>{total_open}</b> open bugs currently unresolved. Prioritize clearing these before shipping the next version.", "risk-medium"),
        ("⚠️ High Severity Rate", f"<b>{pct_high:.1f}%</b> of all bugs are High severity. Introduce stricter pre-release checklists and static code analysis.", "risk-medium"),
        ("✅ Best Quality", f"Version <b>{best_quality_ver}</b> has the highest quality score. Use its branching strategy and test coverage as a baseline.", "risk-low"),
        ("🧪 Test Automation", f"Modules with >20% reopen rates ({most_reopen_mod}) should be covered by automated regression suites in CI/CD.", "risk-medium"),
        ("📊 Data-Driven QA", f"Focus the next sprint on {highest_risk_mod}, {most_reopen_mod}, and any module with >30 open occurrences to maximally reduce release risk.", "risk-high"),
    ]

    for i in range(0, len(insights), 2):
        ic1, ic2 = st.columns(2)
        for col, insight in zip([ic1, ic2], insights[i:i+2]):
            with col:
                label, text, cls = insight
                col.markdown(f"""
                <div class='insight-card {cls}'>
                  <span class='priority-badge'>{label}</span>
                  <div style='margin-top:10px; line-height:1.6;'>{text}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Testing priority list ────────────────────────────────────────────────
    st.markdown("#### 🎯 Testing Priority List — Next Release")
    priority_df = risk_df[["Module", "RiskScore", "RiskLevel", "ReopenRate", "TotalOccurrences"]].copy()
    priority_df["Priority Rank"] = range(1, len(priority_df) + 1)
    priority_df["Recommended Test Coverage"] = priority_df["RiskScore"].apply(
        lambda s: "≥ 90% coverage + automated regression" if s >= 66
        else "≥ 75% coverage + smoke tests" if s >= 33
        else "≥ 60% coverage + exploratory"
    )
    priority_df["ReopenRate"] = priority_df["ReopenRate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(
        priority_df.rename(columns={
            "RiskScore": "Risk Score", "RiskLevel": "Risk Level",
            "TotalOccurrences": "Total Occurrences", "ReopenRate": "Reopen Rate",
        }),
        use_container_width=True, hide_index=True
    )
