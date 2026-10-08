from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from macropulse.crisis import build_crisis_radar
from macropulse.data import CRISIS_TARGETS, FORECAST_START_YEAR, HISTORICAL_END_YEAR, load_dataset
from macropulse.features import engineer_features
from macropulse.portfolio import country_macro_scores, fit_regimes, illustrative_weights

st.set_page_config(page_title="MacroPulse | Global Macro Research", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: #f5f7fb; color: #172136; }
[data-testid="stSidebar"] { background: #111b2c; }
[data-testid="stSidebar"] * { color: #edf3ff !important; }
[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf2; border-radius: 14px; padding: 16px 18px; box-shadow: 0 3px 12px #18294c0a; }
[data-testid="stMetricLabel"] { color: #63718a; }
[data-testid="stMetricValue"] { color: #16243b; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
.pulse-eyebrow { color: #5574f7; text-transform: uppercase; font: 500 11px 'DM Mono', monospace; letter-spacing: .12em; }
.pulse-title { font-size: clamp(32px,4vw,48px); font-weight: 800; letter-spacing: -.05em; color: #142039; line-height: 1.05; margin: 5px 0 6px; }
.pulse-byline { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; color: #34445f; font-size: 13px; margin: 4px 0 8px; }
.pulse-author-email a { color: #4567f5; font-weight: 600; text-decoration: none; }
.pulse-author-email a:hover { text-decoration: underline; }
.pulse-social-links { display: inline-flex; align-items: center; gap: 7px; margin-left: 2px; }
.pulse-social-links a { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border: 1px solid #e1e7f2; border-radius: 50%; background: white; transition: transform .15s ease, box-shadow .15s ease; }
.pulse-social-links a:hover { transform: translateY(-1px); box-shadow: 0 3px 10px #18294c18; }
.pulse-social-icon { display: block; width: 15px; height: 15px; }
.pulse-subtitle { color: #66748a; font-size: 14px; margin-bottom: 20px; }
.pulse-note { border-left: 3px solid #e7ad47; background: #fff9ec; padding: 11px 14px; border-radius: 3px 10px 10px 3px; color: #66532c; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("MACROPULSE_DATA", str(ROOT / "data" / "GMD.csv")))
LINKEDIN_ICON = base64.b64encode((ROOT / "assets" / "linkedin.svg").read_bytes()).decode("ascii")
X_ICON = base64.b64encode((ROOT / "assets" / "x.svg").read_bytes()).decode("ascii")

@st.cache_data(show_spinner=False)
def load_cached(path: str):
    return load_dataset(path)

@st.cache_data(show_spinner=False)
def load_uploaded(contents: bytes):
    return load_dataset(contents)

@st.cache_data(show_spinner="Engineering annual macro features…")
def features_cached(frame: pd.DataFrame):
    return engineer_features(frame)

@st.cache_data(show_spinner="Fitting historical regime model…")
def regimes_cached(frame: pd.DataFrame, cutoff: int):
    return fit_regimes(frame, training_end_year=cutoff)

@st.cache_data(show_spinner="Scoring crisis models and evaluating the temporal holdout…")
def crisis_cached(frame: pd.DataFrame, cutoff: int):
    return build_crisis_radar(frame, as_of_year=cutoff)

st.markdown(f'''
<div class="pulse-eyebrow">GLOBAL MACRO · QUANT RESEARCH</div>
<div class="pulse-title">MacroPulse</div>
<div class="pulse-byline"><strong>Built by Prakhar Gupta</strong>
<span class="pulse-author-email">· <a href="mailto:bestofprakhar@gmail.com">bestofprakhar@gmail.com</a></span>
<span class="pulse-social-links">
<a href="https://www.linkedin.com/in/prakhar-gupta-5b7250372/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn profile" title="LinkedIn">
<img class="pulse-social-icon" src="data:image/svg+xml;base64,{LINKEDIN_ICON}" alt=""></a>
<a href="https://x.com/PrakharQuant" target="_blank" rel="noopener noreferrer" aria-label="X profile" title="X">
<img class="pulse-social-icon" src="data:image/svg+xml;base64,{X_ICON}" alt=""></a>
</span></div>
<div class="pulse-subtitle">Annual macro regimes, country risk signals, and transparent portfolio tilts.</div>
''', unsafe_allow_html=True)

try:
    if DATA_PATH.is_file():
        raw, audit = load_cached(str(DATA_PATH))
    else:
        st.info("For privacy and licensing, the source dataset is not bundled with this app. Upload an authorized copy to analyze it.")
        uploaded = st.file_uploader("Upload GMD.csv", type=["csv"], help="The file is processed in this app session and is not committed to the GitHub repository.")
        if uploaded is None:
            st.stop()
        raw, audit = load_uploaded(uploaded.getvalue())
except Exception as exc:
    st.error(f"Dataset unavailable: {exc}")
    st.markdown("Place the supplied file at `data/GMD.csv`, set the `MACROPULSE_DATA` environment variable, or upload it above. The source CSV is intentionally not included in the repository.")
    st.stop()

if raw.empty:
    st.error("No data is available in the configured analysis window (1960–2031).")
    st.stop()

history_years = raw.loc[raw.year.le(HISTORICAL_END_YEAR), "year"]
if history_years.empty:
    st.error("No historical sample through 2024 is available for analysis.")
    st.stop()
max_history = int(history_years.max())
min_history = int(history_years.min())
with st.sidebar:
    st.markdown("### Research controls")
    analysis_year = st.slider("Historical cutoff", min_value=min_history, max_value=max_history, value=max_history, help="Model fitting and portfolio scoring are restricted to this year. Rows from 2025–2031 are never used as historical crisis outcomes.")
    country_list = sorted(raw["countryname"].dropna().unique().tolist())
    st.caption(f"{len(country_list)} countries · {int(raw.year.min())}–{int(raw.year.max())} in the selected file")
    st.markdown("---")
    st.caption("MacroPulse is a research prototype. It does not use asset prices, represent investable performance, or provide investment advice.")

panel = features_cached(raw)
regimes = regimes_cached(panel, analysis_year)
historical = regimes.loc[regimes.year.le(analysis_year)].copy()
projection = regimes.loc[regimes.year.ge(FORECAST_START_YEAR)].copy()
latest_rows = historical.sort_values(["ISO3", "year"]).groupby("ISO3", as_index=False, sort=False).tail(1)
latest_regime = latest_rows["regime"].value_counts().index[0] if not latest_rows.empty and latest_rows["regime"].notna().any() else "Not available"

st.markdown(f'<div class="pulse-note"><b>Sample discipline:</b> Historical fit through {analysis_year}. Rows dated 2025–2031 are shown separately as forward/forecast-period observations and are excluded from crisis training and holdout outcomes. The dataset does not include source-level provenance or a formal forecast marker for every value.</div>', unsafe_allow_html=True)

t_overview, t_country, t_crisis, t_portfolio, t_audit = st.tabs(["Global overview", "Country lab", "Crisis radar", "Macro tilts", "Data audit"])

with t_overview:
    st.subheader("Global macro state")
    a, b, c, d = st.columns(4)
    a.metric("Countries covered", f"{historical.ISO3.nunique():,}")
    b.metric("Historical observations", f"{len(historical):,}")
    c.metric("Historical window", f"{int(historical.year.min())}–{analysis_year}")
    d.metric("Most common regime", latest_regime)

    trend = historical.groupby("year", as_index=False).agg(
        median_growth=("gdp_growth", "median"), median_inflation=("infl", "median"),
        country_count=("ISO3", "nunique"),
    )
    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.year, y=trend.median_growth, name="Real GDP growth", line=dict(color="#4567f5", width=2.5)))
        fig.add_trace(go.Scatter(x=trend.year, y=trend.median_inflation, name="Inflation", line=dict(color="#e0a53d", width=2.5)))
        fig.add_vrect(x0=FORECAST_START_YEAR - .5, x1=2031.5, fillcolor="#f1b64a", opacity=.09, line_width=0, annotation_text="Forward period", annotation_position="top left")
        fig.update_layout(title="Cross-country median · annual % / source units", height=390, template="plotly_white", hovermode="x unified", margin=dict(l=20,r=20,t=58,b=20), legend=dict(orientation="h", y=-.2))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = latest_rows["regime"].value_counts().rename_axis("Regime description").reset_index(name="Countries")
        fig2 = px.bar(counts, x="Countries", y="Regime description", orientation="h", color="Countries", color_continuous_scale=["#c8d1ff", "#4567f5"], title=f"Latest historical regime labels · {analysis_year}")
        fig2.update_layout(height=390, template="plotly_white", margin=dict(l=15,r=15,t=58,b=15), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    if not projection.empty:
        st.caption(f"The selected CSV contains {len(projection):,} rows dated 2025–{int(projection.year.max())}; these are shown as a forward period, not treated as verified realized history.")

with t_country:
    st.subheader("Country research panel")
    selected_country = st.selectbox("Select economy", country_list, index=country_list.index("India") if "India" in country_list else 0)
    country_data = historical.loc[historical.countryname.eq(selected_country)].sort_values("year")
    if country_data.empty:
        st.info("No observations for this country before the selected cutoff.")
    else:
        row = country_data.iloc[-1]
        growth = row.get("gdp_growth", np.nan)
        infl = row.get("infl", np.nan)
        debt = row.get("govdebt_GDP", np.nan)
        ca = row.get("CA_GDP", np.nan)
        u = row.get("unemp", np.nan)
        cols = st.columns(5)
        for col, label, value, decimals, suffix in [
            (cols[0], "Real GDP growth", growth, 2, "%"), (cols[1], "Inflation", infl, 2, "%"),
            (cols[2], "Debt / GDP", debt, 1, "%"), (cols[3], "Current account / GDP", ca, 1, "%"),
            (cols[4], "Unemployment", u, 1, "%"),
        ]:
            metric_value = "—" if pd.isna(value) else f"{value:.{decimals}f}{suffix}"
            col.metric(label, metric_value)
        st.caption(f"Latest available row: {int(row.year)} · regime label: **{row.get('regime', 'Unavailable')}**. These indicators inherit the source dataset's units and definitions.")
        g1, g2 = st.columns(2)
        with g1:
            fig = px.line(country_data, x="year", y="gdp_growth", markers=True, title="Real GDP growth · log change (%)", color_discrete_sequence=["#4567f5"])
            fig.update_layout(template="plotly_white", height=320, margin=dict(l=15,r=15,t=55,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig = px.line(country_data, x="year", y="infl", markers=True, title="Inflation · source series", color_discrete_sequence=["#df9d32"])
            fig.update_layout(template="plotly_white", height=320, margin=dict(l=15,r=15,t=55,b=20))
            st.plotly_chart(fig, use_container_width=True)
        cols_show = [c for c in ["year", "gdp_growth", "infl", "govdebt_GDP", "govdef_GDP", "CA_GDP", "unemp", "strate", "regime"] if c in country_data]
        st.dataframe(country_data[cols_show].sort_values("year", ascending=False).head(12), use_container_width=True, hide_index=True)

with t_crisis:
    st.subheader("Next-year crisis early warning")
    st.caption("Regularized logistic models use information at year t to score the crisis indicator at t+1. Temporal holdout: train on rows through 2010; evaluate on subsequent labeled years through the selected cutoff. Displayed model scores are exploratory and not independently calibrated.")
    predictions, metrics = crisis_cached(panel, analysis_year)
    metric_labels = {"SovDebtCrisis": "Sovereign debt", "CurrencyCrisis": "Currency", "BankingCrisis": "Banking"}
    mcols = st.columns(3)
    for idx, target in enumerate(CRISIS_TARGETS):
        m = metrics.loc[metrics.crisis.eq(target.replace("Crisis", " crisis"))]
        if m.empty:
            mcols[idx].metric(f"{metric_labels[target]} · ROC-AUC", "—")
            continue
        val = m.iloc[0]["ROC-AUC"]
        mcols[idx].metric(f"{metric_labels[target]} · ROC-AUC", "—" if pd.isna(val) else f"{val:.3f}")
    display_target = st.selectbox("Risk series", CRISIS_TARGETS, format_func=lambda x: metric_labels[x])
    risk = predictions.dropna(subset=[display_target]).sort_values(display_target, ascending=False)
    if risk.empty:
        st.warning("Not enough observed crisis events and macro features to fit this model within the historical sample.")
    else:
        top = risk.head(20).copy()
        top["Model score (%)"] = top[display_target] * 100
        fig = px.bar(top, x="Model score (%)", y="countryname", orientation="h", color="Model score (%)", color_continuous_scale=["#fed984", "#d94d50"], title=f"Highest model scores · next-year {metric_labels[display_target].lower()} crisis")
        fig.update_layout(template="plotly_white", height=520, yaxis={"categoryorder": "total ascending"}, margin=dict(l=15,r=20,t=55,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top[["countryname", "ISO3", "year", "Model score (%)"]], use_container_width=True, hide_index=True)
    with st.expander("Out-of-time model diagnostics"):
        shown = metrics.copy()
        for col in ["ROC-AUC", "PR-AUC", "Brier"]:
            shown[col] = shown[col].map(lambda x: "—" if pd.isna(x) else f"{x:.4f}")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption("ROC-AUC measures ranking, PR-AUC is informative for rare events, and Brier score measures probability error. Sparse/missing crisis labels reduce the effective sample.")

with t_portfolio:
    st.subheader("Country macro tilts")
    st.caption("An equal-weight, cross-sectional heuristic: GDP growth (+), absolute inflation (−), public debt ratio (−), current-account balance (+), and unemployment (−). It is not trained on asset returns and is not a portfolio backtest.")
    scores = country_macro_scores(panel, as_of_year=analysis_year)
    n = st.slider("Countries in illustrative basket", min_value=3, max_value=20, value=10)
    cash = st.slider("Illustrative cash reserve", min_value=0, max_value=30, value=10, step=5) / 100
    weights = illustrative_weights(scores, top_n=n, cash_weight=cash)
    if weights.empty:
        st.warning("Not enough macro coverage to form country scores.")
    else:
        total_weight = weights.weight.sum()
        x1, x2 = st.columns([1.2, 1])
        with x1:
            graph = weights.copy()
            graph["Weight (%)"] = graph.weight * 100
            fig = px.bar(graph, x="Weight (%)", y="countryname", orientation="h", color="macro_score", color_continuous_scale="RdYlGn", title=f"Illustrative macro tilt · {analysis_year}", hover_data={"ISO3": True, "macro_score": ":.2f", "Weight (%)": ":.1f"})
            fig.update_layout(template="plotly_white", height=480, yaxis={"categoryorder": "total ascending"}, margin=dict(l=15,r=20,t=55,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with x2:
            show = weights.copy()
            show["Macro score"] = show.macro_score.map(lambda x: "Cash" if pd.isna(x) else f"{x:+.2f}")
            show["Weight"] = show.weight.map(lambda x: f"{x:.1%}")
            st.dataframe(show[["countryname", "ISO3", "Macro score", "Weight"]], use_container_width=True, hide_index=True, height=450)
            st.metric("Weights sum", f"{total_weight:.1%}")
        st.info("There are no asset-price series in the supplied GMD file. MacroPulse therefore does not report CAGR, Sharpe ratio, volatility, drawdown, transaction costs, or investable country returns.")

with t_audit:
    st.subheader("Data integrity & coverage")
    a, b, c, d = st.columns(4)
    a.metric("Rows loaded for analysis", f"{audit['analysis_rows']:,}")
    b.metric("Countries", f"{audit['countries']:,}")
    c.metric("Rows excluded outside 1960–2031", f"{audit['dropped_outside_window']:,}")
    d.metric("Duplicate country-years removed", f"{audit['dropped_duplicate_country_year']:,}")
    st.caption(f"Input file: `{audit['source_name']}` · {audit['source_bytes'] / 1_000_000:.1f} MB · source rows: {audit['input_rows']:,}")
    coverage = raw.groupby("year", as_index=False).agg(countries=("ISO3", "nunique"), gdp_observed=("rGDP", "count"), inflation_observed=("infl", "count"), currency_labels=("CurrencyCrisis", "count"))
    fig = px.area(coverage, x="year", y="countries", title="Country-year coverage in the selected window", color_discrete_sequence=["#5573f5"])
    fig.add_vrect(x0=FORECAST_START_YEAR - .5, x1=2031.5, fillcolor="#f1b64a", opacity=.12, line_width=0, annotation_text="Forward period", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=15,r=15,t=55,b=15))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(coverage.sort_values("year", ascending=False).head(25), use_container_width=True, hide_index=True)
    st.markdown("**Interpretation guardrails**")
    st.markdown("- Annual macro data supports low-frequency research; it cannot support next-day trading claims.\n- Crisis indicators are sparse and unevenly observed; missing labels are not treated as non-events.\n- Data vintages, units, revisions, and forecast provenance are not supplied in this CSV.\n- Regime names and country scores are exploratory descriptions, not causal conclusions or recommendations.")

st.markdown("---")
st.caption("MacroPulse · Research prototype · No investment advice · Use only with data you are authorized to analyze")
