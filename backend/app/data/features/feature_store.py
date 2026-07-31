import pandas as pd 

from app.data.indicator_registry import FEATURES
from app.data.features.transforms import TRANSFORMS
from app.data.processing.align import (
    build_master_index,
    align_to_index,
    merge_features,
)

from app.data.ingestion.yahoo import load_parquet as load_yahoo
from app.data.ingestion.fred import load_parquet as load_fred
from app.data.ingestion.gpr import load_gpr_parquet as load_gpr


LOADERS = {
    "yahoo": load_yahoo,
    "fred": load_fred,
    "gpr": load_gpr,
}


class FeatureStore:

    def __init__(self):
        self.features = pd.DataFrame()

    def load_indicator(self, indicator: dict) -> pd.DataFrame:

        loader = LOADERS[indicator["loader"]]

        if indicator["loader"] == "yahoo":
            path = f"data/raw/yahoo/{indicator['ticker']}.parquet"

        elif indicator["loader"] == "fred":
            path = f"data/raw/fred/{indicator['series']}.parquet"

        elif indicator["loader"] == "gpr":

            if indicator["frequency"] == "daily":
                path = "data/raw/gpr/daily.parquet"
            else:
                path = "data/raw/gpr/monthly.parquet"

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

        return self.features

    def save(
        self,
        filepath: str = "data/processed/features_v1.parquet"
    ):

        self.features.to_parquet(filepath)