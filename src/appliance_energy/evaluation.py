"""
Forecast evaluation metrics.
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_rmse(
    y_true,
    y_pred
):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def calculate_mase(
    y_true,
    y_pred,
    y_train,
    seasonality=24
):
    """
    Calculate Mean Absolute Scaled Error.

    The denominator uses the in-sample seasonal
    naive forecasting error.
    """
    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    y_train = np.asarray(
        y_train,
        dtype=float
    )

    scale = np.mean(
        np.abs(
            y_train[seasonality:]
            -
            y_train[:-seasonality]
        )
    )

    if scale == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
        / scale
    )


def calculate_bias(
    y_true,
    y_pred
):
    """Calculate mean forecast bias."""
    return np.mean(
        np.asarray(y_pred)
        -
        np.asarray(y_true)
    )


def evaluate_forecast(
    model_name,
    y_true,
    y_pred,
    y_train,
    seasonality=24
):
    """
    Calculate the common forecasting metrics.
    """
    return {
        "Model": model_name,

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": calculate_rmse(
            y_true,
            y_pred
        ),

        "MASE": calculate_mase(
            y_true,
            y_pred,
            y_train,
            seasonality
        ),

        "Bias": calculate_bias(
            y_true,
            y_pred
        ),
    }