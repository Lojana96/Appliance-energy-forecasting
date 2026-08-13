"""
SARIMAX modelling utilities.
"""

from itertools import product

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
)


def search_sarimax_orders(
    y_train,
    X_train=None,
    p_values=range(7),
    d_values=range(3),
    q_values=range(7),
    seasonal_order=(1, 0, 1, 24),
):
    """
    Perform AIC-based SARIMAX parameter search.

    The function does not run unless explicitly called.
    """
    results = []

    combinations = product(
        p_values,
        d_values,
        q_values
    )

    for p, d, q in combinations:

        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted = model.fit(
                disp=False,
                maxiter=100
            )

            results.append({
                "p": p,
                "d": d,
                "q": q,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
            })

        except Exception:
            results.append({
                "p": p,
                "d": d,
                "q": q,
                "AIC": np.nan,
                "BIC": np.nan,
            })

    return (
        pd.DataFrame(results)
        .dropna(subset=["AIC"])
        .sort_values("AIC")
        .reset_index(drop=True)
    )


def fit_sarimax(
    y_train,
    order,
    seasonal_order,
    X_train=None
):
    """Fit the selected SARIMAX model."""
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(
        disp=False,
        maxiter=100
    )


def forecast_sarimax(
    fitted_model,
    horizon,
    X_future=None
):
    """
    Generate SARIMAX forecast and confidence intervals.
    """
    forecast = (
        fitted_model
        .get_forecast(
            steps=horizon,
            exog=X_future
        )
    )

    return (
        forecast.predicted_mean,
        forecast.conf_int()
    )