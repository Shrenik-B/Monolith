"""
ai_analyst.py

Purpose:
--------
AI Market Analyst Synthesis Engine.
Generates automated, highly structured AI macro analysis reports and regime explainability 
based on HMM regime predictions, macro feature drivers, historical crisis detection, and DTW trajectory matches.
"""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd


HISTORICAL_CRISES = [
    {"name": "Global Financial Crisis (GFC / Lehman Collapse)", "start": "2007-12-01", "end": "2009-06-30",
     "desc": "Systemic banking crisis triggered by subprime mortgage defaults, massive deleveraging, liquidity freeze, and historic equity drawdowns."},
    {"name": "US Sovereign Credit Downgrade & European Debt Stress", "start": "2011-07-01", "end": "2011-10-31",
     "desc": "S&P downgraded US AAA credit rating amid Washington debt-ceiling standoff alongside European peripheral sovereign debt contagion."},
    {"name": "China Currency Devaluation & Global Oil Crash", "start": "2015-08-01", "end": "2016-02-28",
     "desc": "Sharp Yuan devaluation, crude oil collapse under $30/bbl, and emerging market capital flight creating global growth panic."},
    {"name": "COVID-19 Global Pandemic Liquidity Crisis", "start": "2020-02-15", "end": "2020-04-30",
     "desc": "Unprecedented global economic lockdowns causing fastest 30% S&P market decline in history and severe treasury market illiquidity."},
    {"name": "Aggressive Fed Tightening & 40-Year Inflation Peak", "start": "2022-01-01", "end": "2022-11-30",
     "desc": "Rapid 500+ bps Federal Reserve interest rate hikes to curb 9%+ headline inflation, triggering simultaneous stock and bond bear market."},
    {"name": "SVB & US Regional Banking Liquidity Shock", "start": "2023-03-01", "end": "2023-05-15",
     "desc": "Rapid bank runs on Silicon Valley Bank and Signature Bank due to duration mismatches, requiring emergency Fed bank funding facility."}
]


def detect_historical_crisis_context(date_str: str) -> str:
    """Checks if date_str falls within a known historical crisis epoch."""
    try:
        dt = pd.to_datetime(date_str)
        for crisis in HISTORICAL_CRISES:
            start_dt = pd.to_datetime(crisis["start"])
            end_dt = pd.to_datetime(crisis["end"])
            if start_dt <= dt <= end_dt:
                return (
                    f"⚠️ **HISTORICAL CRISIS PERIOD DETECTED**: This date falls within the **{crisis['name']}**.\n\n"
                    f"**Crisis Summary**: {crisis['desc']}"
                )
    except Exception:
        pass
    return ""


def generate_ai_market_commentary(
    regime_info: Dict[str, Any],
    macro_row: Dict[str, float],
    dtw_matches: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generates comprehensive AI macro analysis commentary for the selected market date.
    """
    date_str = regime_info.get("date", "Selected Date")
    current_regime = regime_info.get("current_regime_name", "Unknown Regime")
    confidence = regime_info.get("confidence_score", 0.0) * 100
    tomorrow_regime = regime_info.get("tomorrow_regime_name", "Unknown Regime")
    tomorrow_conf = regime_info.get("tomorrow_confidence", 0.0) * 100

    vix = macro_row.get("vix_percentile", 0.5)
    credit_spread = macro_row.get("credit_spread", 2.0)
    yield_curve = macro_row.get("yield_curve", 1.0)
    inflation = macro_row.get("inflation_z", 0.0)
    fedfunds = macro_row.get("fedfunds_z", 0.0)
    spy_ret = macro_row.get("spy_returns", 0.0) * 100

    # 1. Executive Overview & Regime Characterization
    crisis_banner = detect_historical_crisis_context(date_str)
    
    overview = (
        f"As of **{date_str}**, the Gaussian HMM engine classifies the macro environment as **{current_regime}** "
        f"with a statistical model confidence score of **{confidence:.1f}%**.\n\n"
    )

    if crisis_banner:
        overview += f"{crisis_banner}\n\n"

    # Contextual regime explanation
    if "Recessionary" in current_regime or "Bear" in current_regime or vix > 0.7 or credit_spread > 3.5:
        overview += (
            "**Market Interpretation**: Systemic stress is elevated. High equity volatility combined with expanding credit "
            "risk premiums signals heightened recession risk, liquidity constraints, and defensive institutional positioning."
        )
    elif "Goldilocks" in current_regime or "Low Volatility" in current_regime:
        overview += (
            "**Market Interpretation**: Macro conditions reflect stable expansion. Suppressed volatility, manageable inflation, "
            "and healthy credit markets provide a supportive tailwind for risk assets."
        )
    elif "Inflationary" in current_regime or "Late Cycle" in current_regime:
        overview += (
            "**Market Interpretation**: The economy is experiencing late-cycle momentum with elevated price pressures. "
            "Central bank tightening risks and margin compression warrant tactical discipline."
        )
    else:
        overview += (
            "**Market Interpretation**: Macroeconomic features indicate balanced market conditions with steady growth "
            "and moderate cyclical rebalancing."
        )

    # 2. Tomorrow's Outlook & Regime Persistence
    has_next = regime_info.get("has_next_day", False)
    actual_next_date = regime_info.get("actual_next_date")
    actual_next_name = regime_info.get("actual_next_regime_name")
    actual_changed = regime_info.get("actual_regime_changed", False)
    top_trans_name = regime_info.get("top_transition_regime_name", "N/A")
    top_trans_prob = regime_info.get("top_transition_prob", 0.0) * 100

    outlook_parts = []

    if has_next and actual_changed:
        outlook_parts.append(
            f"⚡ **REALIZED HISTORICAL REGIME SHIFT DETECTED ({actual_next_date})**: "
            f"On the next trading day in history (**{actual_next_date}**), the market officially transitioned from "
            f"**{current_regime}** ➔ **{actual_next_name}**."
        )
    elif has_next:
        outlook_parts.append(
            f"🔄 **REALIZED HISTORICAL PERSISTENCE ({actual_next_date})**: "
            f"On the next trading day (**{actual_next_date}**), the regime remained anchored in **{actual_next_name}**."
        )

    outlook_parts.append(
        f"**1-Day Forward Markov Projection**: The model calculates a **{tomorrow_conf:.1f}% probability** "
        f"for **{tomorrow_regime}**. Primary off-diagonal transition candidate target if a shift occurs: **{top_trans_name}** ({top_trans_prob:.1f}% transition probability)."
    )

    outlook = "\n\n".join(outlook_parts)

    # 3. Macro Feature Driver Deep-Dive (Why It Is Happening)
    drivers = []
    
    # Yield Curve analysis
    if yield_curve < 0:
        drivers.append(
            f"• **Yield Curve Inversion (`{yield_curve:.2f}%`)** 🚨: The 10Y-2Y treasury spread is negative, "
            f"a historically reliable lead indicator of economic contraction and credit tightening."
        )
    else:
        drivers.append(
            f"• **Yield Curve Slope (`{yield_curve:.2f}%`)** ✅: Normal positive slope indicating healthy short vs long term rate expectations."
        )

    # Volatility analysis
    vix_pct = vix * 100
    if vix_pct > 70:
        drivers.append(
            f"• **VIX Volatility Percentile (`{vix_pct:.1f}%`)** ⚠️: Implied volatility is in the upper quartile, reflecting institutional panic or hedging demand."
        )
    else:
        drivers.append(
            f"• **VIX Volatility Percentile (`{vix_pct:.1f}%`)**: Subdued market volatility indicating calm trading conditions."
        )

    # Credit Spread analysis
    if credit_spread > 3.0:
        drivers.append(
            f"• **Credit Spread (`{credit_spread:.2f}%`)** ⚠️: High-yield bond risk premium is elevated, signaling corporate borrowing stress."
        )
    else:
        drivers.append(
            f"• **Credit Spread (`{credit_spread:.2f}%`)**: Corporate credit spreads remain tight, reflecting solid liquidity."
        )

    # Inflation & Fed Policy
    drivers.append(f"• **Inflation Z-Score (`{inflation:+.2f} std dev`)**: Deviation of price index from long-run historical baseline.")
    drivers.append(f"• **Fed Funds Z-Score (`{fedfunds:+.2f} std dev`)**: Policy rate stance relative to neutral historical rates.")
    drivers.append(f"• **Daily SPY Return (`{spy_ret:+.2f}%`)**: Direct equity index daily return.")

    drivers_text = "\n\n".join(drivers)

    # 4. Historical Trajectory Analogs (Past Only)
    analogs_summary = []
    for rank, m in enumerate(dtw_matches[:3], 1):
        ret_val = m.get('forward_30d_return')
        if ret_val is not None:
            ret_str = f"{ret_val:+.2f}%"
            ret_desc = "Outperformed (Bullish)" if ret_val > 2.0 else ("Declined (Bearish)" if ret_val < -2.0 else "Flat / Rangebound")
        else:
            ret_str = "N/A"
            ret_desc = "Data unavailable"

        analogs_summary.append(
            f"**Match #{rank} — Period `{m['start_date']}` ➔ `{m['end_date']}`**:\n"
            f"  - **Trajectory Similarity**: `{m['similarity_score']*100:.1f}%` (Normalized DTW Distance: `{m['normalized_distance']:.4f}`)\n"
            f"  - **Historical Outcome (Next 30 Days)**: **{ret_str}** — *{ret_desc}*\n"
            f"  - **Search Policy**: Evaluated strictly from past historical data prior to query window."
        )

    analogs_text = "\n\n".join(analogs_summary) if analogs_summary else "No prior historical trajectory matches found."

    return {
        "overview": overview,
        "outlook": outlook,
        "drivers": drivers_text,
        "analogs": analogs_text,
    }
