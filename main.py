from dotenv import load_dotenv
import polars as pl
import numpy as np
from nixtla import NixtlaClient
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

load_dotenv() # for NIXTLA_API_KEY

nixtla_client = NixtlaClient()
nixtla_client.validate_api_key()


def load_and_prepare_incidence_data(data_path):
    data = pl.read_csv(data_path)
    print(data)

    data = data.with_columns(pl.datetime(pl.col("Year"), 1, 1).alias("ds"))
    data = data.rename({"Council": "unique_id", "Incidence": "y"})
    data = data.select(["unique_id", "ds", "y", "Population"])
    return data


def load_and_prepare_ruc_data(data_path):
    data = pl.read_csv(data_path)
    print(data)

   # set "Urban" to 1 and "Rural" to 0
    data = data.with_columns(
        pl.when(pl.col("Urban_rural_flag") == "Urban").then(1)
        .when(pl.col("Urban_rural_flag") == "Rural").then(0)
        .otherwise(None).alias("is_urban")
    )
    data = data.select(["LAD24NM", "LAD24CD", "is_urban"])
    return data


def create_future_years(data, future_years=2):
    """Create a new DataFrame with future years for forecasting."""
    # create a new dataframe with the same columns but with years 2023 and 2024
    future_years_list = np.arange(2023, 2023 + future_years).tolist()

    is_urban = data.select(["unique_id", "is_urban"]).unique()
    # average population for each unique_id
    population = data.group_by("unique_id").agg(pl.col("Population").mean())

    future_years = pl.DataFrame({
        "unique_id": np.repeat(data["unique_id"].unique().to_list(), future_years),
        "ds": future_years_list * len(data["unique_id"].unique()),
    })

    future_years = future_years.join(is_urban, on="unique_id", how="left")
    future_years = future_years.join(population, on="unique_id", how="left")
    return future_years


def main():
    data_path = "data/UKHSA-2017-2022-Lyme-Disease.csv"
    data = load_and_prepare_incidence_data(data_path)

    # there are some missing due to boundary changes
    ruc_path = "data/rural_urban_classification.csv"
    ruc = load_and_prepare_ruc_data(ruc_path)

    # Merge data with rural urban classification
    data = data.join(ruc, left_on="unique_id", right_on="LAD24NM", how="left")
    data = data.select(["unique_id", "ds", "y", "Population", "is_urban"])
    data = data.filter(pl.col("y").is_not_null())

    horizon = 5  # next 5 years, but clip to 2023 and 2024 when saving

    future_ex_vars_df = create_future_years(data, future_years=horizon)
    print(future_ex_vars_df)

    sf = StatsForecast( 
        models=[ARIMA(order=(1, 1, 1))], # Damped-trend linear Exponential smoothing, no seasonality
        freq=1,
        n_jobs=-1
    )

    # need ds column as year int from timestamp
    data_arima = data.with_columns(pl.col("ds").dt.year().cast(pl.Int32).alias("ds"))
    data_arima = data_arima.select(["unique_id", "ds", "y"])

    arima_forecasts = sf.forecast(df=data_arima, level=[95], h=horizon)

    arima_forecasts = arima_forecasts.with_columns(
        pl.when(pl.col("ARIMA") < 0).then(0).otherwise(pl.col("ARIMA")).alias("ARIMA"),
        pl.when(pl.col("ARIMA-lo-95") < 0).then(0).otherwise(pl.col("ARIMA-lo-95")).alias("ARIMA-lo-95"),
        pl.when(pl.col("ARIMA-hi-95") < 0).then(0).otherwise(pl.col("ARIMA-hi-95")).alias("ARIMA-hi-95"),
    )

    arima_forecasts = arima_forecasts.with_columns(pl.datetime(pl.col("ds"), 1, 1).alias("ds"))

    print(arima_forecasts.describe())

    forecasts = nixtla_client.forecast(
        df=data,
        X_df=future_ex_vars_df,
        # level=[95],
        h=horizon,
        freq="1y",
        time_col="ds",
        target_col="y"
    )

    # clip
    forecasts = forecasts.with_columns(pl.when(pl.col("TimeGPT") < 0).then(0).otherwise(pl.col("TimeGPT")).alias("TimeGPT"))
    # create TimeGPT-lo-95 and TimeGPT-hi-95 which are just ARIMA-lo-95 and ARIMA-hi-95 from a different df
    forecasts = forecasts.join(
        arima_forecasts.select(["unique_id", "ds", "ARIMA-lo-95", "ARIMA-hi-95"]),
        on=["unique_id", "ds"],
        how="left"
    )
    print(forecasts)

    fig = nixtla_client.plot(data, forecasts)
    fig.savefig("submission/forecast.png", bbox_inches="tight")

    # filter for ds 2023 and 2024
    # and convert to schema Year,Council,Incidence,Lower_95CI,Upper_95CI
    result = forecasts.with_columns(
        pl.col("ds").dt.year().cast(pl.Int32).alias("Year"),
        pl.col("unique_id").alias("Council"),
        pl.col("TimeGPT").alias("Incidence"),
        pl.col("ARIMA-lo-95").alias("Lower_95CI"),
        pl.col("ARIMA-hi-95").alias("Upper_95CI"),
    ).select([
        "Year", "Council", "Incidence", "Lower_95CI", "Upper_95CI"
    ]).filter(
        pl.col("Year").is_in([2023, 2024])
    ).sort("Year")
    print(result)

    # write csv
    result.write_csv("submission/forecast.csv", include_header=True)


if __name__ == "__main__":
    main()
