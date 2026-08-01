"""
app.py

Purpose:
--------
Production Web Dashboard for Market Regime Detection & DTW Historical Trajectory Similarity.
Version 1 Architecture: K-Means++ Init ➔ Baum–Welch Gaussian HMM ➔ AIC/BIC Model Selection ➔ DTW Engine ➔ AI Analyst
"""

import importlib
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from backend.app.quant.regime import load, predict
from backend.app.quant.analogue import dtw_engine
from backend.app.services import ai_analyst
from backend.app.core import config

# Dev-only module auto-reload gate
if os.getenv("APP_ENV") == "dev":
    importlib.reload(load)
    importlib.reload(predict)
    importlib.reload(dtw_engine)
    importlib.reload(ai_analyst)

from backend.app.quant.regime.load import retrieve_features
from backend.app.quant.regime.predict import HMMPredictor, get_regime_name
from backend.app.quant.analogue.dtw_engine import DTWEngine
from backend.app.services.ai_analyst import generate_ai_market_commentary

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Market Regime Intelligence | V1 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e0f7fa; font-size: 2rem; margin: 0; }
    .main-header p { color: #90caf9; font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    .regime-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stMetric label { font-size: 0.78rem; font-weight: 600; }
    div[data-testid="stExpander"] { border: 1px solid #1e3a5f; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📈 Market Regime Intelligence Dashboard</h1>
    <p>Version 1 · K-Means++ Init → Gaussian HMM (Baum–Welch) → AIC/BIC Model Selection → DTW Historical Search → AI Analyst</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# COLOR MAP FOR REGIMES
# ==============================================================================
REGIME_COLORS = {
    "Goldilocks Expansion (Low Inflation)":         "#2ecc71",
    "Low Volatility Bull Market":                   "#27ae60",
    "Steady Growth / Mid-Cycle Expansion":          "#1abc9c",
    "Inflationary Bull / Growth Peak":              "#16a085",
    "Inflationary Expansion / Cyclical Peak":       "#f39c12",
    "Late Cycle / Inflationary Growth":             "#e67e22",
    "Market Consolidation / Neutral Growth":        "#3498db",
    "Sideways / Range-Bound Market":               "#2980b9",
    "Stagflationary Bear / Rate-Hike Stress":       "#e74c3c",
    "Recessionary Bear Market / Volatile Stress":   "#c0392b",
    "Recessionary Bear Market / High Stress":       "#922b21",
    "Transitional / Uncertain Regime":              "#95a5a6",
}

def regime_color(name: str) -> str:
    return REGIME_COLORS.get(name, "#95a5a6")

# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_resource(show_spinner="⚙️ Loading pipeline data and model...")
def load_pipeline_data():
    X, feature_names, df_clean = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    predictor_inst = HMMPredictor()
    df_regimes = predictor_inst.get_regime_dataframe(X, df_clean)
    if "spy_returns" in df_regimes.columns:
        df_regimes["cumulative_market_index"] = (1 + df_regimes["spy_returns"]).cumprod() * 100
    else:
        df_regimes["cumulative_market_index"] = 100.0
    df_regimes["rolling_30d_return"] = df_regimes["spy_returns"].rolling(30).sum() * 100 if "spy_returns" in df_regimes.columns else 0.0
    return X, feature_names, df_clean, df_regimes, predictor_inst

try:
    X, feature_names, df_clean, df_regimes, predictor = load_pipeline_data()
except Exception as e:
    st.error(f"❌ Error loading pipeline artifacts: {e}. Please run `python pipeline.py` first.")
    st.stop()

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("## 🗓️ Market Date Selection")
st.sidebar.markdown("Pick **ANY trading date** from 2003 to 2026.")

min_available_date = df_clean.index.min().date()
max_available_date = df_clean.index.max().date()

selected_calendar_date = st.sidebar.date_input(
    "Select Target Market Date:",
    value=max_available_date,
    min_value=min_available_date,
    max_value=max_available_date
)

selected_date_str = selected_calendar_date.strftime("%Y-%m-%d")

st.sidebar.divider()
st.sidebar.markdown("### 🤖 Model Summary")
meta = predictor.metadata
st.sidebar.metric("HMM Hidden States", meta.get("n_states", "N/A"))
st.sidebar.metric("Training Observations", f"{meta['metrics'].get('n_samples', len(df_clean)):,}" if 'n_samples' in meta.get('metrics', {}) else f"{len(df_clean):,}")
st.sidebar.metric("AIC", f"{meta['metrics']['aic']:,.0f}")
st.sidebar.metric("BIC", f"{meta['metrics']['bic']:,.0f}")
st.sidebar.metric("Log-Likelihood", f"{meta['metrics']['log_likelihood']:,.1f}")
st.sidebar.metric("Features", str(meta.get("n_features", len(feature_names))))
trained_at = meta.get("training_timestamp", "")[:10]
st.sidebar.caption(f"🕐 Model trained: `{trained_at}`")

st.sidebar.divider()
st.sidebar.markdown("### 📐 Regime Legend")
mapping = predictor.regime_mapping or {}
for state_id, regime_nm in sorted(mapping.items()):
    color = regime_color(regime_nm)
    st.sidebar.markdown(f"<span style='color:{color}; font-size:1.1rem;'>●</span> **State {state_id}** — {regime_nm}", unsafe_allow_html=True)

# ==============================================================================
# INFERENCE FOR SELECTED DATE
# ==============================================================================
date_info = predictor.predict_for_date(X, dates=df_clean.index, target_date_str=selected_date_str, df_clean=df_clean)
target_idx = date_info["target_idx"]
macro_row = df_clean.iloc[target_idx].to_dict()
current_mapping = date_info.get("regime_mapping", {})

# Regime color for current regime
curr_regime_color = regime_color(date_info["current_regime_name"])

# ==============================================================================
# TOP METRICS
# ==============================================================================
st.markdown(f"### 📊 Market Intelligence Summary — `{date_info['date']}`")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("🏷️ Current Market Regime", date_info["current_regime_name"],
              help="Gaussian HMM detected macro regime.")
with m2:
    st.metric("🎯 Model Confidence", f"{date_info['confidence_score'] * 100:.1f}%",
              help="Posterior probability score — how certain the model is.")
with m3:
    if date_info.get("has_next_day") and date_info.get("actual_regime_changed"):
        next_reg_name = date_info["actual_next_regime_name"]
        next_date = date_info["actual_next_date"]
        st.metric(
            f"⚡ Realized Shift ({next_date})",
            next_reg_name,
            delta=f"State shifted on {next_date}",
            delta_color="inverse",
            help=f"Realized ground-truth regime transition on next trading day ({next_date}) in historical dataset."
        )
    elif date_info.get("has_next_day"):
        next_reg_name = date_info["actual_next_regime_name"]
        next_date = date_info["actual_next_date"]
        st.metric(
            f"🔄 Realized Regime ({next_date})",
            next_reg_name,
            delta=f"Persisted on {next_date}",
            help=f"Realized ground-truth regime outcome on next trading day ({next_date}) in historical dataset."
        )
    else:
        st.metric(
            "🔮 Tomorrow's Forecast",
            date_info["tomorrow_regime_name"],
            delta=f"{date_info['tomorrow_confidence'] * 100:.1f}% prob",
            help="Markov chain 1-day forward transition forecast."
        )
with m4:
    vix_pct = macro_row.get("vix_percentile", 0.0) * 100
    vix_label = "🔴 Elevated Fear" if vix_pct > 70 else ("🟡 Moderate" if vix_pct > 40 else "🟢 Calm")
    st.metric("😨 VIX Percentile", f"{vix_pct:.1f}%", delta=vix_label,
              help="Market fear index rank vs 2003–2026 history.")
with m5:
    cs = macro_row.get("credit_spread", 0.0)
    cs_label = "🔴 Wide Stress" if cs > 3.0 else ("🟡 Watch" if cs > 2.5 else "🟢 Tight")
    st.metric("📉 Credit Spread", f"{cs:.2f}%", delta=cs_label,
              help="HY corporate bond spread — credit market health.")

# Educational Guide
with st.expander("📖 What are Market Regimes? — Educational Guide", expanded=False):
    st.markdown("""
### 💡 What are Market Regimes?
Financial markets move through distinct statistical economic phases (**Regimes**) driven by inflation, volatility, growth, credit, and monetary policy.
A Gaussian Hidden Markov Model (HMM) detects these phases by learning which combinations of 17 macroeconomic features tend to cluster together across 5,525 trading days of history (2003–2026).

#### 🏷️ The 6 Economic Market Regimes:
| Regime | Characteristics | Example Period |
|---|---|---|
| **🟢 Goldilocks Expansion** | Low inflation, stable growth, tight credit | 2012–2014, 2016–2017 |
| **🟢 Low Volatility Bull Market** | Suppressed VIX, positive returns, calm macro | 2017, 2019 |
| **🟡 Inflationary Expansion** | Strong growth but rising inflation & rates | 2021 H1, 2018 |
| **🟠 Late Cycle / Inflationary Growth** | Slowing growth under rate pressure | 2006–2007, 2018 Q4 |
| **🔴 Recessionary Bear Market / Volatile Stress** | High VIX, wide credit spreads, sell-offs | 2008 GFC, 2020 COVID |
| **⚫ Recessionary Bear Market / High Stress** | Extreme tail-risk, rare systemic crisis | Extreme intraday panic days |

#### 📊 Key Stress Indicators:
- **Yield Curve (10Y–2Y spread)**: Inversion (negative) signals recession risk 12–18 months ahead.
- **VIX Percentile > 70**: Institutional panic / hedging demand.
- **Credit Spread > 3.0%**: Corporate borrowing under stress — potential liquidity freeze.
- **DTW Engine**: Scans **only past historical data** to find similar market episodes and what happened next.
""")

# Pre-compute shared DTW matches for DTW & AI Analyst tabs (avoids cross-tab dependency)
dtw_engine_obj = DTWEngine(window_size=30)
spy_vals = df_clean["spy_returns"].values if "spy_returns" in df_clean.columns else None
q_start_idx_shared = max(0, target_idx - 30)
q_start_date_shared = df_clean.index[q_start_idx_shared].strftime("%Y-%m-%d")

shared_matches = dtw_engine_obj.find_similar_trajectories(
    X=X,
    dates=df_clean.index,
    regimes=df_regimes["predicted_regime"].values,
    top_n=3,
    query_start_date=q_start_date_shared,
    query_end_date=selected_date_str,
    spy_returns=spy_vals
)

st.divider()

# ==============================================================================
# MAIN TABS
# ==============================================================================
tab_overview, tab_forecast, tab_dtw, tab_ai, tab_diagnostics = st.tabs([
    "📊 Regime & Market Charts",
    "🔮 Tomorrow's Forecast",
    "🔍 DTW Trajectory Search",
    "🤖 AI Market Analyst",
    "⚙️ Macro Drivers & Model"
])

# ==============================================================================
# TAB 1: MARKET REGIME & TREND CHARTS
# ==============================================================================
with tab_overview:
    st.markdown("#### 📈 Cumulative Market Growth (Base 100, 2003–Present)")
    st.caption("Each dot color represents the HMM-detected market regime on that trading day. The red dashed line marks your selected date.")

    # Regime color sequence aligned to actual regime names in data
    unique_regimes = sorted(df_regimes["regime_name"].unique())
    color_seq = [regime_color(r) for r in unique_regimes]

    # Chart 1: Cumulative Market Index colored by regime
    fig_market = px.scatter(
        df_regimes.reset_index(),
        x="Date",
        y="cumulative_market_index",
        color="regime_name",
        color_discrete_sequence=color_seq,
        color_discrete_map={r: regime_color(r) for r in unique_regimes},
        labels={"regime_name": "Economic Regime", "cumulative_market_index": "Market Index (Base 100)"},
        render_mode="svg"
    )
    fig_market.update_traces(marker=dict(size=4, opacity=0.85))
    fig_market.add_vline(x=date_info['date'], line_width=2, line_dash="dash", line_color="white",
                         annotation_text=f"Selected: {date_info['date']}", annotation_position="top right")
    fig_market.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e0e0e0", legend_title_text="Regime",
        xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
    )
    st.plotly_chart(fig_market, use_container_width=True)

    # Chart 2: 30-Day Rolling SPY Return
    st.markdown("#### 📉 30-Day Rolling Market Return (%)")
    fig_rolling = px.area(
        df_regimes.reset_index(),
        x="Date",
        y="rolling_30d_return",
        color_discrete_sequence=["#3498db"],
        labels={"rolling_30d_return": "30-Day Cumulative Return (%)"}
    )
    fig_rolling.add_hline(y=0, line_color="white", line_width=1)
    fig_rolling.add_vline(x=date_info['date'], line_width=2, line_dash="dash", line_color="red")
    fig_rolling.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e0e0e0", height=280,
        xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    # Chart 3: Interactive Indicator selector
    st.markdown("#### 🔬 Interactive Macro Indicator Explorer")
    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        selected_indicator = st.selectbox(
            "Select Macro Feature to Plot:",
            options=feature_names,
            index=feature_names.index("vix_percentile") if "vix_percentile" in feature_names else 0,
            key="macro_selector"
        )
    with sel_col2:
        show_regime_bands = st.checkbox("Color by Regime", value=True, key="regime_bands")

    if show_regime_bands:
        fig_ind = px.scatter(
            df_regimes.reset_index(),
            x="Date",
            y=selected_indicator,
            color="regime_name",
            color_discrete_map={r: regime_color(r) for r in unique_regimes},
            labels={"regime_name": "Economic Regime"},
            render_mode="svg"
        )
    else:
        fig_ind = px.line(df_regimes.reset_index(), x="Date", y=selected_indicator, color_discrete_sequence=["#3498db"])

    fig_ind.update_traces(marker=dict(size=3, opacity=0.75))
    fig_ind.add_vline(x=date_info['date'], line_width=2, line_dash="dash", line_color="red")
    fig_ind.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0",
        xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    # Charts 4 & 5: Probability bar + Regime pie
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### 🎯 Regime Probabilities on `{date_info['date']}`")
        state_probs_df = pd.DataFrame({
            "Economic Regime": list(date_info["state_probabilities"].keys()),
            "Probability (%)": [v * 100 for v in date_info["state_probabilities"].values()]
        }).sort_values("Probability (%)", ascending=False)
        fig_probs = px.bar(
            state_probs_df,
            x="Economic Regime", y="Probability (%)",
            text="Probability (%)",
            color="Economic Regime",
            color_discrete_map={r: regime_color(r) for r in state_probs_df["Economic Regime"]}
        )
        fig_probs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_probs.update_layout(
            yaxis_range=[0, 115], showlegend=False,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0",
            xaxis_tickangle=-30, xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_probs, use_container_width=True)

    with col_b:
        st.markdown("#### 🥧 Historical Regime Frequency (2003–2026)")
        regime_freq = df_regimes["regime_name"].value_counts().reset_index()
        regime_freq.columns = ["Economic Regime", "Days Count"]
        fig_pie = px.pie(
            regime_freq, values="Days Count", names="Economic Regime",
            hole=0.45, color="Economic Regime",
            color_discrete_map={r: regime_color(r) for r in regime_freq["Economic Regime"]}
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
        fig_pie.update_layout(
            paper_bgcolor="#0e1117", font_color="#e0e0e0",
            legend=dict(orientation="v", font=dict(size=10))
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# TAB 2: TOMORROW'S REGIME FORECAST
# ==============================================================================
with tab_forecast:
    st.markdown(f"### 🔮 Next-Day Market Regime Intelligence — from `{date_info['date']}`")

    # 1. Ground Truth Realized Next-Day Outcome (for historical dates)
    if date_info.get("has_next_day"):
        actual_date = date_info["actual_next_date"]
        actual_name = date_info["actual_next_regime_name"]
        actual_changed = date_info["actual_regime_changed"]
        act_color = regime_color(actual_name)

        if actual_changed:
            st.markdown(f"""
<div style='background-color:#2c1919; border: 2px solid #e74c3c; border-left: 8px solid #e74c3c; padding:1.2rem 1.5rem; border-radius:10px; margin-bottom:1.5rem;'>
    <strong style='font-size:1.15rem; color:#e74c3c;'>⚡ REALIZED HISTORICAL REGIME SHIFT ON {actual_date}</strong><br/>
    <span style='font-size:1.25rem; color:#ffffff;'>State transitioned from <strong>{date_info['current_regime_name']}</strong> ➔ <strong style='color:{act_color};'>{actual_name}</strong></span><br/>
    <span style='color:#bdc3c7; font-size:0.9rem;'>Ground-truth outcome recorded in dataset on next trading day ({actual_date}).</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style='background-color:#19271d; border: 1px solid #2ecc71; border-left: 6px solid #2ecc71; padding:1rem 1.5rem; border-radius:10px; margin-bottom:1.5rem;'>
    <strong style='font-size:1.05rem; color:#2ecc71;'>🔄 REALIZED HISTORICAL REGIME PERSISTENCE ON {actual_date}</strong><br/>
    <span style='font-size:1.15rem; color:#ffffff;'>Regime remained <strong>{actual_name}</strong> on the next trading day.</span>
</div>
""", unsafe_allow_html=True)

    # 2. Markov Chain Forward Projection Engine
    tomorrow_name = date_info["tomorrow_regime_name"]
    tomorrow_conf = date_info["tomorrow_confidence"] * 100
    t_color = regime_color(tomorrow_name)
    persist = date_info["current_regime_name"] == tomorrow_name

    top_trans_name = date_info.get("top_transition_regime_name", "N/A")
    top_trans_prob = date_info.get("top_transition_prob", 0.0) * 100
    trans_out_prob = date_info.get("transition_out_prob", 0.0) * 100

    st.markdown(f"""
<div style='background-color:#1a2a3a; border-left: 5px solid {t_color}; padding:1rem 1.5rem; border-radius:8px; margin-bottom:1.5rem;'>
    <strong style='font-size:1.1rem; color:{t_color};'>{'🔄 Markov Chain Projecting Regime Persistence' if persist else '⚡ Markov Transition Warning'}</strong><br/>
    <span style='font-size:1.25rem; color:#e0e0e0;'>1-Day Forward Markov Projection: <strong>{tomorrow_name}</strong> ({tomorrow_conf:.1f}% confidence)</span><br/>
    <span style='color:#90caf9; font-size:0.92rem;'>Daily State Persistence: <strong>{date_info.get('persistence_prob', 0)*100:.1f}%</strong> | Transition Shift Hazard Rate: <strong>{trans_out_prob:.1f}%</strong></span><br/>
    <span style='color:#f39c12; font-size:0.92rem;'>🎯 Primary Candidate Target If Shift Occurs: <strong>{top_trans_name}</strong> ({top_trans_prob:.1f}% transition probability)</span>
</div>
""", unsafe_allow_html=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("#### Tomorrow's Full Probability Distribution")
        tomorrow_df = pd.DataFrame({
            "Target Economic Regime": list(date_info["tomorrow_probabilities"].keys()),
            "Probability (%)": [v * 100 for v in date_info["tomorrow_probabilities"].values()]
        }).sort_values("Probability (%)", ascending=False)
        fig_tomorrow = px.bar(
            tomorrow_df,
            x="Target Economic Regime", y="Probability (%)",
            text="Probability (%)",
            color="Target Economic Regime",
            color_discrete_map={r: regime_color(r) for r in tomorrow_df["Target Economic Regime"]}
        )
        fig_tomorrow.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_tomorrow.update_layout(
            yaxis_range=[0, 115], showlegend=False,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0",
            xaxis_tickangle=-30, xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_tomorrow, use_container_width=True)

    with f_col2:
        st.markdown("#### State Transition Probability Heatmap")
        trans_matrix = predictor.model.transmat_ * 100
        n_states = predictor.n_states
        # Use dynamic mapping labels if available
        states_labels = [current_mapping.get(i, get_regime_name(i)) for i in range(n_states)]
        short_labels = [s.split("/")[0].strip()[:22] for s in states_labels]

        fig_heatmap = px.imshow(
            trans_matrix,
            x=short_labels,
            y=short_labels,
            labels=dict(x="Tomorrow's Regime", y="Today's Regime", color="Probability (%)"),
            text_auto=".1f",
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_heatmap.update_layout(
            height=420, paper_bgcolor="#0e1117", font_color="#e0e0e0",
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Regime persistence explanation
    st.markdown("#### 📐 Daily State Persistence (Diagonal of Transition Matrix)")
    tm = predictor.model.transmat_
    persist_data = [{"Regime": current_mapping.get(s, get_regime_name(s)), 
                     "Daily Persistence (%)": tm[s, s] * 100,
                     "Avg Days in Regime": round(1 / (1 - tm[s, s])) if tm[s, s] < 1.0 else 999}
                    for s in range(predictor.n_states)]
    persist_df = pd.DataFrame(persist_data).sort_values("Daily Persistence (%)", ascending=False)
    st.dataframe(persist_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: DTW TRAJECTORY ALIGNMENT
# ==============================================================================
with tab_dtw:
    st.markdown("### 🔍 Dynamic Time Warping Historical Trajectory Search")
    st.info("🔒 **Strict Past-Only Policy**: The DTW engine exclusively searches historical market windows that ended **before** the query window start date. Zero future data contamination.")

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        dtw_window_size = st.slider("Trajectory Window Size (Trading Days)", min_value=15, max_value=90, value=30, step=5, key="dtw_window")
    with w_col2:
        dtw_top_k = st.slider("Number of Top Historical Matches", min_value=1, max_value=5, value=3, key="dtw_top_k")

    q_start_idx = max(0, target_idx - dtw_window_size)
    q_start_date_str = df_clean.index[q_start_idx].strftime("%Y-%m-%d")

    with st.spinner("⏳ Computing DTW trajectory matches..."):
        dtw_engine_obj = DTWEngine(window_size=dtw_window_size)
        spy_vals = df_clean["spy_returns"].values if "spy_returns" in df_clean.columns else None
        matches = dtw_engine_obj.find_similar_trajectories(
            X=X,
            dates=df_clean.index,
            regimes=df_regimes["predicted_regime"].values,
            top_n=dtw_top_k,
            query_start_date=q_start_date_str,
            query_end_date=selected_date_str,
            spy_returns=spy_vals
        )

    st.markdown(f"**Query Window**: `{q_start_date_str}` ➔ `{selected_date_str}` ({dtw_window_size} trading days) | **Found {len(matches)} matches**")
    st.divider()

    # Multi-feature trajectory comparison
    st.markdown("#### 📈 VIX Trajectory Alignment (Query vs Historical Matches)")
    fig_dtw_lines = go.Figure()
    query_vix = df_clean.iloc[q_start_idx: target_idx + 1]["vix_percentile"].values
    fig_dtw_lines.add_trace(go.Scatter(
        y=query_vix, mode="lines+markers",
        name=f"📍 Query Window → {selected_date_str}",
        line=dict(color="#e74c3c", width=3)
    ))
    colors_dtw = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for rank, m in enumerate(matches, 1):
        hist_vix = df_clean.iloc[m['start_idx']: m['end_idx']]["vix_percentile"].values
        fig_dtw_lines.add_trace(go.Scatter(
            y=hist_vix, mode="lines",
            name=f"Match #{rank} [{m['start_date']} → {m['end_date']}]",
            line=dict(color=colors_dtw[rank - 1], dash="dash", width=1.5)
        ))
    fig_dtw_lines.update_layout(
        xaxis_title="Day in Window Sequence",
        yaxis_title="VIX Percentile",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0",
        xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
    )
    st.plotly_chart(fig_dtw_lines, use_container_width=True)

    # Historical match cards
    st.markdown("#### 🏅 Top Historical Market Period Matches")
    for rank, m in enumerate(matches, 1):
        regime_nm = current_mapping.get(m['dominant_regime'], get_regime_name(m['dominant_regime'])) if m['dominant_regime'] is not None else "N/A"
        r_color = regime_color(regime_nm)
        ret_val = m.get('forward_30d_return')
        ret_str = f"{ret_val:+.2f}%" if ret_val is not None else "N/A"
        ret_color = "#2ecc71" if (ret_val or 0) > 0 else "#e74c3c"

        st.markdown(f"""
<div style='border:1px solid {colors_dtw[rank-1]}; border-radius:10px; padding:1rem; margin-bottom:0.5rem; background:#0e1c2a;'>
    <strong style='font-size:1.05rem; color:{colors_dtw[rank-1]};'>Match #{rank}</strong>
    <span style='color:#90caf9; margin-left:1rem;'>Period: <code>{m['start_date']}</code> → <code>{m['end_date']}</code></span>
</div>
""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Similarity Score", f"{m['similarity_score']*100:.2f}%")
        with c2:
            st.metric("DTW Distance", f"{m['normalized_distance']:.4f}")
        with c3:
            st.metric("Historical Regime", regime_nm)
        with c4:
            st.metric("Next 30-Day Outcome", ret_str)
        st.divider()


# ==============================================================================
# TAB 4: AI MARKET ANALYST REPORT
# ==============================================================================
with tab_ai:
    st.markdown(f"### 🤖 AI Market Analyst Executive Report — `{date_info['date']}`")
    st.caption("Automated macro synthesis from HMM regime parameters, macro feature drivers, historical crisis detection, and DTW trajectory matches.")

    ai_report_key = f"ai_report_{selected_date_str}"
    
    col_btn, col_info = st.columns([1.5, 3.5])
    with col_btn:
        generate_clicked = st.button("🤖 Generate AI Commentary", key=f"btn_ai_{selected_date_str}")
    
    dtw_input_matches = shared_matches if 'shared_matches' in locals() else (matches if 'matches' in locals() else [])

    if generate_clicked or ai_report_key in st.session_state:
        if generate_clicked or ai_report_key not in st.session_state:
            with st.spinner("🧠 Generating macro executive commentary..."):
                st.session_state[ai_report_key] = generate_ai_market_commentary(
                    regime_info=date_info,
                    macro_row=macro_row,
                    dtw_matches=dtw_input_matches
                )
        
        ai_report = st.session_state[ai_report_key]

        ai_c1, ai_c2 = st.columns([2, 1])
        with ai_c1:
            st.markdown("#### 📌 Executive Market Overview")
            st.info(ai_report["overview"])

            st.markdown("#### 🔮 Tomorrow's Outlook & Regime Persistence")
            st.success(ai_report["outlook"])

            st.markdown("#### 📜 Historical Analogs & Precedent Analysis")
            st.warning(ai_report["analogs"])

        with ai_c2:
            st.markdown("#### 📊 Macroeconomic Feature Drivers")
            st.markdown(ai_report["drivers"])

            # Radar / spider chart for key macro features
            st.markdown("#### 🕸️ Macro Risk Radar")
            radar_features = ["vix_percentile", "credit_spread", "yield_curve", "inflation_z", "fedfunds_z"]
            radar_labels = ["VIX Fear", "Credit Spread", "Yield Curve", "Inflation", "Fed Funds"]
            radar_vals = []
            for f in radar_features:
                v = macro_row.get(f, 0.0)
                # Normalize each feature to 0–1 range for radar display
                col_vals = df_clean[f].values if f in df_clean.columns else np.array([0.0])
                v_min, v_max = col_vals.min(), col_vals.max()
                v_norm = (v - v_min) / (v_max - v_min + 1e-9)
                radar_vals.append(float(np.clip(v_norm, 0, 1)))
            
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(231,76,60,0.25)",
                line=dict(color="#e74c3c", width=2)
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                paper_bgcolor="#0e1117", font_color="#e0e0e0",
                margin=dict(l=20, r=20, t=20, b=20), height=320
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info(f"💡 Click **Generate AI Commentary** above to run the automated AI Analyst report for **{date_info['date']}**.")


# ==============================================================================
# TAB 5: MACRO DRIVERS & MODEL METADATA
# ==============================================================================
with tab_diagnostics:
    st.markdown(f"#### 🔬 Full Macroeconomic Feature Vector — `{date_info['date']}`")

    feature_keys = [k for k in macro_row.keys() if k in feature_names]
    feature_vals = [macro_row[k] for k in feature_keys]
    feat_df = pd.DataFrame({"Macro Feature": feature_keys, "Value": feature_vals}).sort_values("Value", ascending=False)

    colors_feat = ["#e74c3c" if v > 1.0 else ("#f39c12" if v > 0.5 else ("#2ecc71" if v < -0.5 else "#3498db")) for v in feat_df["Value"]]
    fig_feat = go.Figure(go.Bar(
        x=feat_df["Macro Feature"], y=feat_df["Value"],
        marker_color=colors_feat, text=[f"{v:.3f}" for v in feat_df["Value"]],
        textposition="outside"
    ))
    fig_feat.add_hline(y=0, line_color="white", line_width=1)
    fig_feat.update_layout(
        title=f"Macro Feature Vector on {date_info['date']} (Standardized)",
        xaxis_tickangle=-45, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e0e0e0", xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("#### 📋 Dataset Statistics")
        st.markdown(f"- **Total Observations**: {len(df_clean):,} trading days (2003–2026)")
        st.markdown(f"- **Total Features**: {len(feature_names)} macroeconomic & financial features")
        st.markdown(f"- **Date Range**: `{df_clean.index.min().date()}` to `{df_clean.index.max().date()}`")
        st.markdown(f"- **Model**: Gaussian HMM, {meta.get('n_states')} states, Full covariance")
        st.markdown(f"- **Training EM Iterations**: {meta['metrics'].get('iterations', 'N/A')}")
        st.markdown(f"- **Convergence**: {'✅ Yes' if meta['metrics'].get('converged') else '❌ No'}")

    with d_col2:
        st.markdown("#### 🏷️ Dynamic Empirical Regime Mapping")
        for state_id, regime_nm in sorted(current_mapping.items()):
            color = regime_color(regime_nm)
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>● State {state_id}</span> — {regime_nm}", unsafe_allow_html=True)

    with st.expander("📂 Raw Feature Values for Selected Date"):
        st.json({k: round(v, 6) if isinstance(v, float) else v for k, v in macro_row.items()})

    with st.expander("📂 HMM Model Metadata"):
        st.json(predictor.metadata)
