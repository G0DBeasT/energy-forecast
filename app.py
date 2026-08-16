"""
Energy Grid Load Forecasting Dashboard.

Interactive, production-grade Streamlit web application providing three clearly separated
operational forecasting products:
1. 🌅 Next-Day Forecast (24 Hours / Hourly Dispatch)
2. 📅 Next-Week Forecast (7 Days / Weekly Planning & Reserve Sizing)
3. 🗓️ Next-Month Forecast (30 Days / Long-Range Resource Budgeting)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    DATA_PROC,
    TARGET_COL,
    TEMP_COL,
    HUMIDITY_COL,
    WEATHER_SOURCE_COL,
    DATE_COL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
)
from src.models.registry import ModelRegistry
from src.forecast.engine import MultiHorizonForecastEngine

# Page Setup
st.set_page_config(
    page_title="Energy Grid Load Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design Tokens & UI Styles
st.markdown("""
<style>
    .stApp {
        background-color: #0c1017;
        color: #e6edf3;
    }
    .kpi-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .kpi-label {
        color: #8b949e;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }
    .kpi-val {
        color: #f0f6fc;
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .kpi-sub {
        color: #3fb950;
        font-size: 13px;
        font-weight: 500;
        margin-top: 2px;
    }
    .kpi-sub-alert {
        color: #f85149;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-sim {
        background-color: #1f242c;
        border: 1px solid #388bfd;
        color: #58a6ff;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-nwp {
        background-color: #122818;
        border: 1px solid #238636;
        color: #3fb950;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 11px;
    }
    .badge-clim {
        background-color: #2b2210;
        border: 1px solid #d29922;
        color: #e3b341;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_artifacts():
    """Load historical series, metrics summaries, and forecast products."""
    actual_path = DATA_PROC / "hourly_demand.parquet"
    metrics_path = DATA_PROC / "metrics_summary.parquet"
    feat_path = DATA_PROC / "features.parquet"

    day_path = DATA_PROC / "forecast_next_day.csv"
    week_h_path = DATA_PROC / "forecast_next_week.csv"
    week_d_path = DATA_PROC / "forecast_next_week_daily.csv"
    month_path = DATA_PROC / "forecast_next_month.csv"

    actual_df = pd.read_parquet(actual_path) if actual_path.exists() else None
    metrics_df = pd.read_parquet(metrics_path) if metrics_path.exists() else None
    feat_df = pd.read_parquet(feat_path) if feat_path.exists() else None

    df_day = pd.read_csv(day_path, parse_dates=[DATE_COL]) if day_path.exists() else None
    df_week_h = pd.read_csv(week_h_path, parse_dates=[DATE_COL]) if week_h_path.exists() else None
    df_week_d = pd.read_csv(week_d_path, parse_dates=[DATE_COL]) if week_d_path.exists() else None
    df_month = pd.read_csv(month_path, parse_dates=[DATE_COL]) if month_path.exists() else None

    # Load champion model
    model = None
    meta = {}
    model_path = DATA_PROC / "best_model.pkl"
    if model_path.exists():
        try:
            model, meta = ModelRegistry.load_model(model_path)
        except Exception:
            pass

    return actual_df, metrics_df, feat_df, df_day, df_week_h, df_week_d, df_month, model, meta


actual_df, metrics_df, feat_df, df_day, df_week_h, df_week_d, df_month, model, meta = load_data_artifacts()

# App Header
st.title("⚡ Energy Grid Demand Forecasting System")
st.caption(f"Multi-Horizon Transmission Load Forecasting & Planning | Region: {LATITUDE}°N, {LONGITUDE}°E ({TIMEZONE})")

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel & Parameters")

st.sidebar.markdown(
    """
    **Data Provenance**:  
    <span class="badge-sim">SIMULATED / BENCHMARK DATA</span>  
    *Weather: Open-Meteo API (Historical Reanalysis & NWP Forecast)*  
    *Demand: Physics-grounded thermodynamic load model*
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

history_window_days = st.sidebar.slider("Historical Viewport (Days)", min_value=3, max_value=30, value=7, step=1)
confidence_level = st.sidebar.selectbox("Confidence Band Level", options=["90% (Standard)", "80% (Moderate)", "95% (Conservative)"], index=0)
bound_col_lower = "lower_bound_90" if "90%" in confidence_level else ("lower_bound_80" if "80%" in confidence_level else "lower_bound_95")
bound_col_upper = "upper_bound_90" if "90%" in confidence_level else ("upper_bound_80" if "80%" in confidence_level else "upper_bound_95")

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Scenario Stress Testing")
temp_anomaly = st.sidebar.slider(
    "Temperature Anomaly Delta (ΔT °C)",
    min_value=-5.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
    help="Simulate severe heatwave (+ΔT) or cold snap (-ΔT) and observe dynamic grid response.",
)

if temp_anomaly != 0.0:
    st.sidebar.info(f"Simulating **{'+' if temp_anomaly > 0 else ''}{temp_anomaly}°C** temperature anomaly on future grid load.")

# Check if data exists
if actual_df is None or df_day is None:
    st.warning("⚠️ Data and model artifacts not found! Please run the pipeline first using `make data`, `make features`, `make train`, `make forecast`.")
    st.stop()

# Interactive Re-calculation if scenario anomaly is applied
if temp_anomaly != 0.0:
    with st.spinner("Re-computing multi-step forecast under simulated temperature anomaly..."):
        engine = MultiHorizonForecastEngine()
        df_day = engine.forecast_next_day(temp_delta=temp_anomaly)
        df_week_h, df_week_d = engine.forecast_next_week(temp_delta=temp_anomaly)
        df_month = engine.forecast_next_month(temp_delta=temp_anomaly)

recent_history = actual_df.tail(history_window_days * 24)

# Three Dedicated Horizon Tabs + Diagnostics + Export
tab_day, tab_week, tab_month, tab_bench, tab_export = st.tabs([
    "🌅 1. Next-Day Forecast (24h)",
    "📅 2. Next-Week Forecast (7 Days)",
    "🗓️ 3. Next-Month Forecast (30 Days)",
    "📊 Model Leaderboard & Diagnostics",
    "📥 Data Export",
])

# ==============================================================================
# TAB 1: NEXT-DAY FORECAST (24 Hours / Hourly Operational Dispatch)
# ==============================================================================
with tab_day:
    st.subheader("🌅 Next-Day 24-Hour Hourly Dispatch Forecast")
    st.caption("High-resolution day-ahead load curve for operational unit commitment, ramp-rate management, and peaking plant dispatch.")

    # Day KPIs
    peak_mw = float(df_day["forecast_mw"].max())
    peak_time = df_day.loc[df_day["forecast_mw"].idxmax(), DATE_COL]
    trough_mw = float(df_day["forecast_mw"].min())
    trough_time = df_day.loc[df_day["forecast_mw"].idxmin(), DATE_COL]
    avg_mw = float(df_day["forecast_mw"].mean())
    total_day_gwh = float(df_day["forecast_mw"].sum() / 1000.0)

    # Ramp rate (MW/hr change)
    df_day_calc = df_day.copy()
    df_day_calc["ramp_mw_hr"] = df_day_calc["forecast_mw"].diff().fillna(0.0)
    max_ramp_up = float(df_day_calc["ramp_mw_hr"].max())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Day-Ahead Peak Load</div>
            <div class="kpi-val">{peak_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub-alert">Peak at {pd.to_datetime(peak_time).strftime('%H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Minimum Night Valley</div>
            <div class="kpi-val">{trough_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub">Trough at {pd.to_datetime(trough_time).strftime('%H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Average Day Demand</div>
            <div class="kpi-val">{avg_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub">Load Factor: {(avg_mw/peak_mw):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Total Daily Energy</div>
            <div class="kpi-val">{total_day_gwh:,.2f} <span style="font-size:16px;">GWh</span></div>
            <div class="kpi-sub">24-Hour Accumulated</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Max Ramp-Up Rate</div>
            <div class="kpi-val">+{max_ramp_up:,.1f} <span style="font-size:16px;">MW/h</span></div>
            <div class="kpi-sub">Thermal Dispatch Req.</div>
        </div>
        """, unsafe_allow_html=True)

    # Next-Day Plot
    fig_day = go.Figure()

    # Recent Actuals (past 48h for immediate context)
    recent_48h = actual_df.tail(48)
    fig_day.add_trace(go.Scatter(
        x=recent_48h.index,
        y=recent_48h[TARGET_COL],
        name="Recent Actual Demand (MW)",
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="%{x}<br><b>Actual: %{y:,.1f} MW</b>",
    ))

    # Day-Ahead Forecast Line
    fig_day.add_trace(go.Scatter(
        x=df_day[DATE_COL],
        y=df_day["forecast_mw"],
        name="Next-Day Forecast (MW)",
        line=dict(color="#f0883e", width=3, dash="dash"),
        hovertemplate="%{x}<br><b>Forecast: %{y:,.1f} MW</b>",
    ))

    # Prediction Interval
    if bound_col_lower in df_day.columns and bound_col_upper in df_day.columns:
        fig_day.add_trace(go.Scatter(
            x=df_day[DATE_COL].tolist() + df_day[DATE_COL].tolist()[::-1],
            y=df_day[bound_col_upper].tolist() + df_day[bound_col_lower].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(240, 136, 62, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"Uncertainty Band ({confidence_level.split()[0]})",
        ))

    fig_day.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Datetime",
        yaxis_title="Power Demand (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_day, use_container_width=True)

    # Sub-row: Hourly Ramp Rate & Weather Correlation
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("⚡ Hourly Dispatch Ramp Rate (MW/hr)")
        fig_ramp = px.bar(
            df_day_calc,
            x=DATE_COL,
            y="ramp_mw_hr",
            color="ramp_mw_hr",
            color_continuous_scale="RdYlGn_r",
            labels={"ramp_mw_hr": "Ramp Rate (MW/h)"},
        )
        fig_ramp.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_ramp, use_container_width=True)

    with col_d2:
        st.subheader("🌡️ Hourly Temperature vs Demand Profile")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df_day[DATE_COL], y=df_day[TEMP_COL], name="Temperature (°C)", line=dict(color="#ff7b72", width=2)))
        fig_t.add_trace(go.Scatter(x=df_day[DATE_COL], y=df_day[HUMIDITY_COL], name="Humidity (%)", line=dict(color="#79c0ff", width=2), yaxis="y2"))
        fig_t.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_t, use_container_width=True)


# ==============================================================================
# TAB 2: NEXT-WEEK FORECAST (7 Days / 168 Hours / Weekly Planning)
# ==============================================================================
with tab_week:
    st.subheader("📅 Next-Week 7-Day Forecasting & Reserve Planning")
    st.caption("Multi-day aggregated forecast and hourly profile for weekly maintenance windows, thermal unit scheduling, and fuel procurement.")

    # Week KPIs
    week_peak_mw = float(df_week_d["peak_mw"].max())
    peak_day_name = df_week_d.loc[df_week_d["peak_mw"].idxmax(), "day_of_week"]
    week_min_mw = float(df_week_d["min_mw"].min())
    min_day_name = df_week_d.loc[df_week_d["min_mw"].idxmin(), "day_of_week"]
    total_week_gwh = float(df_week_d["total_gwh"].sum())

    # Weekday vs Weekend average
    weekday_avg = float(df_week_d[df_week_d["is_weekend"] == 0]["forecast_mean_mw"].mean())
    weekend_avg = float(df_week_d[df_week_d["is_weekend"] == 1]["forecast_mean_mw"].mean())

    # Largest day-over-day change
    max_dod_idx = df_week_d["day_over_day_change_mw"].abs().idxmax()
    max_dod_val = df_week_d.loc[max_dod_idx, "day_over_day_change_mw"]
    max_dod_day = df_week_d.loc[max_dod_idx, "day_of_week"]

    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Weekly Peak Demand</div>
            <div class="kpi-val">{week_peak_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub-alert">Highest on {peak_day_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Weekly Minimum Load</div>
            <div class="kpi-val">{week_min_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub">Lowest on {min_day_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">7-Day Total Volume</div>
            <div class="kpi-val">{total_week_gwh:,.1f} <span style="font-size:16px;">GWh</span></div>
            <div class="kpi-sub">Total Grid Energy</div>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Weekday vs Weekend</div>
            <div class="kpi-val">{weekday_avg:,.0f} / {weekend_avg:,.0f} <span style="font-size:13px;">MW</span></div>
            <div class="kpi-sub">Weekend Delta: {((weekend_avg-weekday_avg)/weekday_avg):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with w5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Max Day-over-Day Shift</div>
            <div class="kpi-val">{'+' if max_dod_val > 0 else ''}{max_dod_val:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub">{max_dod_day} Ramping</div>
        </div>
        """, unsafe_allow_html=True)

    # 7-Day Interactive Hourly Profile
    fig_week = go.Figure()

    fig_week.add_trace(go.Scatter(
        x=recent_history.index,
        y=recent_history[TARGET_COL],
        name="Recent Actuals (MW)",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="%{x}<br><b>Actual: %{y:,.1f} MW</b>",
    ))

    fig_week.add_trace(go.Scatter(
        x=df_week_h[DATE_COL],
        y=df_week_h["forecast_mw"],
        name="7-Day Hourly Forecast (MW)",
        line=dict(color="#f0883e", width=2.5, dash="dash"),
        hovertemplate="%{x}<br><b>Forecast: %{y:,.1f} MW</b>",
    ))

    # Add 90% confidence envelope
    if bound_col_lower in df_week_h.columns and bound_col_upper in df_week_h.columns:
        fig_week.add_trace(go.Scatter(
            x=df_week_h[DATE_COL].tolist() + df_week_h[DATE_COL].tolist()[::-1],
            y=df_week_h[bound_col_upper].tolist() + df_week_h[bound_col_lower].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(240, 136, 62, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Band",
        ))

    fig_week.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Datetime",
        yaxis_title="Power Demand (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_week, use_container_width=True)

    # Day-by-Day Peak & Total Energy Breakdown
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("📊 Daily Peak vs Base Demand (MW)")
        fig_d_bar = go.Figure()
        fig_d_bar.add_trace(go.Bar(x=df_week_d["day_of_week"], y=df_week_d["peak_mw"], name="Daily Peak (MW)", marker_color="#f85149"))
        fig_d_bar.add_trace(go.Bar(x=df_week_d["day_of_week"], y=df_week_d["forecast_mean_mw"], name="Daily Average (MW)", marker_color="#58a6ff"))
        fig_d_bar.add_trace(go.Bar(x=df_week_d["day_of_week"], y=df_week_d["min_mw"], name="Daily Minimum (MW)", marker_color="#3fb950"))
        fig_d_bar.update_layout(template="plotly_dark", barmode="group", height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_d_bar, use_container_width=True)

    with col_w2:
        st.subheader("📋 7-Day Day-by-Day Planning Table")
        st.dataframe(
            df_week_d[["datetime", "day_of_week", "forecast_mean_mw", "peak_mw", "min_mw", "total_gwh", "temp_c", "day_over_day_change_mw"]],
            use_container_width=True,
            height=280,
        )


# ==============================================================================
# TAB 3: NEXT-MONTH FORECAST (30 Days / Long-Range Planning)
# ==============================================================================
with tab_month:
    st.subheader("🗓️ Next-Month 30-Day Energy Trajectory")
    st.caption("Dedicated Daily LightGBM model integrating live 14-day NWP weather forecasts with 16-30 day seasonal climatology.")

    # Month KPIs
    month_total_gwh = float(df_month["total_daily_gwh"].sum())
    month_peak_mw = float(df_month["forecast_mw"].max())
    month_peak_date = df_month.loc[df_month["forecast_mw"].idxmax(), DATE_COL]
    month_avg_mw = float(df_month["forecast_mw"].mean())
    month_avg_temp = float(df_month[TEMP_COL].mean())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">30-Day Total Energy Budget</div>
            <div class="kpi-val">{month_total_gwh:,.1f} <span style="font-size:16px;">GWh</span></div>
            <div class="kpi-sub">{(month_total_gwh / 1000.0):.2f} TWh Volume</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Monthly Expected Peak</div>
            <div class="kpi-val">{month_peak_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub-alert">Expected on {pd.to_datetime(month_peak_date).strftime('%b %d')}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">30-Day Mean Demand</div>
            <div class="kpi-val">{month_avg_mw:,.1f} <span style="font-size:16px;">MW</span></div>
            <div class="kpi-sub">Baseline Operational Load</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Monthly Mean Temperature</div>
            <div class="kpi-val">{month_avg_temp:,.1f} <span style="font-size:16px;">°C</span></div>
            <div class="kpi-sub">Thermal Driver</div>
        </div>
        """, unsafe_allow_html=True)

    # 30-Day Demand Trajectory
    fig_month = go.Figure()

    # Split into Live Forecast vs Climatology segments for visual transparency
    nwp_mask = df_month[WEATHER_SOURCE_COL] == "nwp_forecast"
    df_nwp = df_month[nwp_mask]
    df_clim = df_month[~nwp_mask]

    fig_month.add_trace(go.Scatter(
        x=df_month[DATE_COL],
        y=df_month["forecast_mw"],
        name="30-Day Daily Forecast (MW)",
        line=dict(color="#2ea043", width=3),
        hovertemplate="%{x}<br><b>Daily Demand: %{y:,.1f} MW</b>",
    ))

    # Lead-time dynamic confidence intervals
    if bound_col_lower in df_month.columns and bound_col_upper in df_month.columns:
        fig_month.add_trace(go.Scatter(
            x=df_month[DATE_COL].tolist() + df_month[DATE_COL].tolist()[::-1],
            y=df_month[bound_col_upper].tolist() + df_month[bound_col_lower].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(46, 160, 67, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Uncertainty Growth Band",
        ))

    fig_month.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Daily Average Grid Demand (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Sub-row: Weekly Aggregation & Meteorological Evolution
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("📊 30-Day Aggregated Weekly Energy Volume (GWh)")
        df_month_weekly = df_month.resample("W", on=DATE_COL)["total_daily_gwh"].sum().reset_index()
        df_month_weekly["Week"] = [f"Week {i+1}" for i in range(len(df_month_weekly))]
        fig_w_gwh = px.bar(df_month_weekly, x="Week", y="total_daily_gwh", color="total_daily_gwh", color_continuous_scale="Teal", labels={"total_daily_gwh": "Volume (GWh)"})
        fig_w_gwh.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_w_gwh, use_container_width=True)

    with col_m2:
        st.subheader("🌡️ 30-Day Projected Weather Driver Progression")
        fig_mw = go.Figure()
        fig_mw.add_trace(go.Scatter(x=df_month[DATE_COL], y=df_month[TEMP_COL], name="Daily Temp (°C)", line=dict(color="#ffa657", width=2)))
        fig_mw.add_trace(go.Scatter(x=df_month[DATE_COL], y=df_month[HUMIDITY_COL], name="Daily Humidity (%)", line=dict(color="#79c0ff", width=2), yaxis="y2"))
        fig_mw.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_mw, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL LEADERBOARD & DIAGNOSTICS
# ==============================================================================
with tab_bench:
    st.subheader("📊 Multi-Model Cross-Validation Leaderboard (Phase 8 & 11)")
    st.caption("Chronological 5-fold expanding window validation benchmark across candidate baseline and gradient boosting architectures.")

    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Metrics summary table generated during training pipeline.")

    st.markdown("---")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.subheader("🎯 Feature Importance (Top 15 Drivers)")
        if model is not None and hasattr(model, "feature_importances_") and feat_df is not None:
            feat_names = [c for c in feat_df.columns if c != TARGET_COL]
            importances = model.feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=True).tail(15)
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
            fig_fi.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_f2:
        st.subheader("🛡️ Architecture & Verification Status")
        st.markdown(f"""
        - **Champion Hourly Architecture**: `LightGBM Regressor (Tuned)`
        - **Champion Daily Architecture**: `Dedicated Daily LightGBM`
        - **Cross-Validation**: 5-Fold Chronological `TimeSeriesSplit`
        - **Forecast Horizons**: 24h (Next-Day), 168h (Next-Week), 720h (Next-Month)
        - **Weather Source**: Open-Meteo NWP Forecast (Days 1–14) + Climatology (Days 15–30)
        - **Uncertainty Model**: Horizon-Dependent Uncertainty Scaling
        - **Unit Tests**: Full `pytest` verification suite
        """)


# ==============================================================================
# TAB 5: DATA EXPORT
# ==============================================================================
with tab_export:
    st.subheader("📥 Export Multi-Horizon Forecast Datasets")
    st.caption("Download production-ready forecast artifacts formatted with confidence bounds and meteorological inputs.")

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown("### 🌅 Next-Day Forecast (24h)")
        st.dataframe(df_day.head(5), use_container_width=True)
        st.download_button(
            "📥 Download Next-Day CSV",
            data=df_day.to_csv(index=False).encode("utf-8"),
            file_name="forecast_next_day_24h.csv",
            mime="text/csv",
        )

    with exp2:
        st.markdown("### 📅 Next-Week Forecast (7d)")
        st.dataframe(df_week_d.head(5), use_container_width=True)
        st.download_button(
            "📥 Download Next-Week CSV",
            data=df_week_d.to_csv(index=False).encode("utf-8"),
            file_name="forecast_next_week_7d.csv",
            mime="text/csv",
        )

    with exp3:
        st.markdown("### 🗓️ Next-Month Forecast (30d)")
        st.dataframe(df_month.head(5), use_container_width=True)
        st.download_button(
            "📥 Download Next-Month CSV",
            data=df_month.to_csv(index=False).encode("utf-8"),
            file_name="forecast_next_month_30d.csv",
            mime="text/csv",
        )
