from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.data.indicator_registry import FEATURES

from app.data.ingestion.yahoo import (
    download_asset,
    save_parquet as save_yahoo,
)

from app.data.ingestion.fred import (
    download_fred_series,
    save_fred_parquet as save_fred,
)

from app.data.ingestion.gpr import (
    download_gpr_monthly,
    download_gpr_daily,
    save_gpr_parquet,
)


START_DATE = "1990-01-01"
END_DATE = "2026-12-31"


def build_yahoo():

    print("Downloading Yahoo indicators...")

    for indicator in FEATURES.values():

        if indicator["loader"] != "yahoo":
            continue

        ticker = indicator["ticker"]

        print(f"Downloading {ticker}")

        df = download_asset(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        filepath = Path(f"data/raw/yahoo/{ticker}.parquet")

        save_yahoo(df, filepath)


def build_fred():

    print("Downloading FRED indicators...")

    for indicator in FEATURES.values():

        if indicator["loader"] != "fred":
            continue

        series = indicator["series"]

        print(f"Downloading {series}...")

        try:
            df = download_fred_series(
                series_id=series,
                start_date=START_DATE,
                end_date=END_DATE,
            )

            filepath = Path(f"data/raw/fred/{series}.parquet")
            save_fred(df, filepath)

        except Exception as e:
            print(f"FAILED: {series}")
            print(e)

    # for indicator in FEATURES.values():

    #     if indicator["loader"] != "fred":
    #         continue

    #     series = indicator["series"]

    #     print(f"Downloading {series}")

    #     df = download_fred_series(
    #         series_id=series,
    #         start_date=START_DATE,
    #         end_date=END_DATE,
    #     )

    #     filepath = Path(f"data/raw/fred/{series}.parquet")

    #     save_fred(df, filepath)
    


def build_gpr():

    print("Downloading GPR Monthly")

    monthly = download_gpr_monthly()

    save_gpr_parquet(
        monthly,
        "data/raw/gpr/monthly.parquet",
    )

    print("Downloading GPR Daily")

    daily = download_gpr_daily()

    save_gpr_parquet(
        daily,
        "data/raw/gpr/daily.parquet",
    )


def main():

    build_yahoo()

    build_fred()

    build_gpr()

    print("\nRaw data build complete.")


if __name__ == "__main__":
    main()