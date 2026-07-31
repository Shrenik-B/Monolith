"""
Feature Registry

Defines every raw indicator used by the Feature Store.

The FeatureStore is responsible for:
1. Loading the raw data.
2. Applying the listed pipeline transformations.
3. Renaming the final output column.
"""

FEATURES = {

    # ======================================================
    # MARKET
    # ======================================================

    "spy": {
        "loader": "yahoo",
        "ticker": "SPY",
        "column": "Close",
        "frequency": "daily",
        "pipeline": [
            ("pct_change", {})
        ],
        "output": "spy_returns"
    },

    "gold": {
        "loader": "yahoo",
        "ticker": "GC=F",
        "column": "Close",
        "frequency": "daily",
        "pipeline": [
            ("pct_change", {})
        ],
        "output": "gold_returns"
    },

    "dxy": {
        "loader": "yahoo",
        "ticker": "DX-Y.NYB",
        "column": "Close",
        "frequency": "daily",
        "pipeline": [
            ("pct_change", {})
        ],
        "output": "dxy_returns"
    },

    "oil": {
        "loader": "yahoo",
        "ticker": "CL=F",
        "column": "Close",
        "frequency": "daily",
        "pipeline": [
            ("pct_change", {})
        ],
        "output": "oil_returns"
    },

    "vix": {
        "loader": "yahoo",
        "ticker": "^VIX",
        "column": "Close",
        "frequency": "daily",
        "pipeline": [
            ("rolling_percentile", {"window": 2520})
        ],
        "output": "vix_percentile"
    },

    # ======================================================
    # INFLATION
    # ======================================================

    "cpi": {
        "loader": "fred",
        "series": "CPIAUCSL",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "inflation_z"
    },

    "core_cpi": {
        "loader": "fred",
        "series": "CPILFESL",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "core_inflation_z"
    },

    # ======================================================
    # GROWTH
    # ======================================================

    "industrial_production": {
        "loader": "fred",
        "series": "INDPRO",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "industrial_production_z"
    },

    "retail_sales": {
        "loader": "fred",
        "series": "RSAFS",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "retail_sales_z"
    },

    # "pmi": {
    #     "loader": "fred",
    #     "series": "NAPM",
    #     "frequency": "monthly",
    #     "pipeline": [
    #         ("rolling_zscore", {"window": 120})
    #     ],
    #     "output": "pmi_z"
    # },

    # ======================================================
    # LABOUR
    # ======================================================

    "unemployment": {
        "loader": "fred",
        "series": "UNRATE",
        "frequency": "monthly",
        "pipeline": [
            ("rolling_zscore", {"window": 120})
        ],
        "output": "unemployment_z"
    },

    "payrolls": {
        "loader": "fred",
        "series": "PAYEMS",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "payrolls_z"
    },

    # ======================================================
    # HOUSING
    # ======================================================

    "housing_starts": {
        "loader": "fred",
        "series": "HOUST",
        "frequency": "monthly",
        "pipeline": [
            ("yoy_change", {}),
            ("rolling_zscore", {"window": 120})
        ],
        "output": "housing_z"
    },

    # ======================================================
    # MONETARY POLICY
    # ======================================================

    "fed_funds": {
        "loader": "fred",
        "series": "FEDFUNDS",
        "frequency": "monthly",
        "pipeline": [
            ("rolling_zscore", {"window": 120})
        ],
        "output": "fedfunds_z"
    },

    # ======================================================
    # RATES
    # ======================================================

    "10y": {
        "loader": "fred",
        "series": "DGS10",
        "frequency": "daily",
        "pipeline": [],
        "output": "10y"
    },

    "2y": {
        "loader": "fred",
        "series": "DGS2",
        "frequency": "daily",
        "pipeline": [],
        "output": "2y"
    },

    "baa": {
        "loader": "fred",
        "series": "BAA",
        "frequency": "daily",
        "pipeline": [],
        "output": "baa"
    },

    # ======================================================
    # SENTIMENT / RISK
    # ======================================================

    # "epu": {
    #     "loader": "epu",
    #     "frequency": "monthly",
    #     "pipeline": [
    #         ("rolling_zscore", {"window": 120})
    #     ],
    #     "output": "epu_z"
    # },

    "gpr_monthly": {
        "loader": "gpr",
        "frequency": "monthly",
        "column": "GPR",
        "pipeline": [
            ("rolling_zscore", {"window": 120})
        ],
        "output": "gpr_monthly_z"
    },

    "gpr_daily": {
        "loader": "gpr",
        "frequency": "daily",
        "column": "GPRD",
        "pipeline": [
            ("rolling_zscore", {"window": 2520})
        ],
        "output": "gpr_daily_z"
    }

}