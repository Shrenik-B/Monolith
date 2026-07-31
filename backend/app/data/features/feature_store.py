import pandas as pd 

from app.data.indicator_registry import FEATURES
from app.data.features.transforms import TRANSFORMS
from app.data.processing.align import (
    build_master_index,
    align_to_index,
    merge_features,
)

from app.data.ingestion.yahoo import load_parquet as load_yahoo
from app.data.ingestion.fred import load_fred_parquet as load_fred
from app.data.ingestion.gpr import load_gpr_parquet as load_gpr


LOADERS = {
    "yahoo": load_yahoo,
    "fred": load_fred,
    "gpr": load_gpr,
}

from app.paths import RAW_DATA_DIR
from app.paths import PROCESSED_DATA_DIR


class FeatureStore:

    def __init__(self):
        self.features = pd.DataFrame()

    def load_indicator(self, indicator: dict) -> pd.DataFrame:

        loader = LOADERS[indicator["loader"]]

        if indicator["loader"] == "yahoo":
            path = RAW_DATA_DIR / "yahoo" / f"{indicator['ticker']}.parquet"

        elif indicator["loader"] == "fred":
            path = RAW_DATA_DIR / "fred" / f"{indicator['series']}.parquet"

        elif indicator["loader"] == "gpr":

            if indicator["frequency"] == "daily":
                path = RAW_DATA_DIR / "gpr" / "daily.parquet"
            else:
                path = RAW_DATA_DIR / "gpr" / "monthly.parquet"

        else:
            raise ValueError("Unknown loader.")

        df = loader(path)

        if "column" in indicator:
            df = df[[indicator["column"]]]

        return df

    def apply_pipeline(
        self,
        df: pd.DataFrame,
        pipeline: list
    ) -> pd.DataFrame:

        for transform_name, kwargs in pipeline:

            transform = TRANSFORMS[transform_name]

            df = transform(df, **kwargs)

        return df

    def build(
        self,
        master_indicator: str = "spy"
    ) -> pd.DataFrame:

        master = self.load_indicator(FEATURES[master_indicator])

        master_index = build_master_index(master)

        feature_frames = []
        
        for indicator in FEATURES.values():

            df = self.load_indicator(indicator)

            df = align_to_index(df, master_index)

            df = self.apply_pipeline(
                df,
                indicator["pipeline"]
            )

            df.columns = [indicator["output"]]

            feature_frames.append(df)

        self.features = merge_features(feature_frames)
        # Composite features
        self.features["yield_curve"] = (
            self.features["10y"] - self.features["2y"]
        )

        self.features["credit_spread"] = (
            self.features["baa"] - self.features["10y"]
        )

        # Optional: remove raw columns if Engineer 2 doesn't need them
        self.features = self.features.drop(
            columns=["10y", "2y", "baa"]
        )

        # Remove all rows containing NaNs
        self.features = self.features.dropna()

        return self.features

    def save(self):

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        filepath = PROCESSED_DATA_DIR / "features_v1.parquet"

        self.features.to_parquet(filepath)

        print(f"Saved to {filepath}")