"""
Leakage-safe feature engineering utilities.
"""

import numpy as np


def add_time_features(data):
    """
    Add calendar and cyclical time features.
    """
    result = data.copy()

    result["hour"] = result.index.hour
    result["dayofweek"] = (
        result.index.dayofweek
    )

    result["is_weekend"] = (
        result["dayofweek"] >= 5
    ).astype(int)

    result["hour_sin"] = np.sin(
        2 * np.pi
        * result["hour"]
        / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi
        * result["hour"]
        / 24
    )

    result["dow_sin"] = np.sin(
        2 * np.pi
        * result["dayofweek"]
        / 7
    )

    result["dow_cos"] = np.cos(
        2 * np.pi
        * result["dayofweek"]
        / 7
    )

    return result


def add_lag_features(
    data,
    target="Appliances",
    lags=(1, 24, 168)
):
    """
    Add historical target-value features.
    """
    result = data.copy()

    for lag in lags:
        result[f"lag_{lag}"] = (
            result[target]
            .shift(lag)
        )

    return result


def add_rolling_features(
    data,
    target="Appliances",
    windows=(24,)
):
    """
    Add leakage-safe rolling statistics.

    The target is shifted before rolling operations
    so the current target cannot predict itself.
    """
    result = data.copy()

    historical_target = (
        result[target]
        .shift(1)
    )

    for window in windows:

        result[
            f"roll_mean_{window}"
        ] = (
            historical_target
            .rolling(window)
            .mean()
        )

        result[
            f"roll_std_{window}"
        ] = (
            historical_target
            .rolling(window)
            .std()
        )

    return result


def build_feature_table(
    data,
    target="Appliances"
):
    """
    Construct the final machine-learning feature table.
    """
    result = add_time_features(data)

    result = add_lag_features(
        result,
        target=target
    )

    result = add_rolling_features(
        result,
        target=target
    )

    return result.dropna()