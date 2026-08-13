"""
Benchmark forecasting models.
"""

import numpy as np
import pandas as pd


def mean_forecast(
    train,
    horizon,
    forecast_index
):
    """Forecast using the historical mean."""
    return pd.Series(
        np.repeat(
            train.mean(),
            horizon
        ),
        index=forecast_index
    )


def naive_forecast(
    train,
    horizon,
    forecast_index
):
    """Forecast using the latest observation."""
    return pd.Series(
        np.repeat(
            train.iloc[-1],
            horizon
        ),
        index=forecast_index
    )


def seasonal_naive_forecast(
    train,
    horizon,
    forecast_index,
    seasonal_period
):
    """
    Forecast using observations from the previous
    seasonal period.
    """
    pattern = (
        train
        .iloc[-seasonal_period:]
        .values
    )

    predictions = np.resize(
        pattern,
        horizon
    )

    return pd.Series(
        predictions,
        index=forecast_index
    )


def drift_forecast(
    train,
    horizon,
    forecast_index
):
    """Forecast using a linear drift from first to last observation."""
    n = len(train)

    slope = (
        train.iloc[-1]
        - train.iloc[0]
    ) / (n - 1)

    steps = np.arange(
        1,
        horizon + 1
    )

    predictions = (
        train.iloc[-1]
        + steps * slope
    )

    return pd.Series(
        predictions,
        index=forecast_index
    )