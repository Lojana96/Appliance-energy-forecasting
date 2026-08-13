"""
Chronos foundation-model forecasting utilities.
"""

import torch


def load_chronos_model(
    model_name="amazon/chronos-bolt-tiny"
):
    """
    Load Chronos-Bolt foundation model.
    """
    from chronos import BaseChronosPipeline

    pipeline = (
        BaseChronosPipeline
        .from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    )

    return pipeline


def forecast_chronos(
    pipeline,
    context,
    prediction_length=24,
):
    """
    Generate a zero-shot Chronos forecast.

    Returns the median (0.5 quantile) forecast.
    """
    context_tensor = torch.tensor(
        context,
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = pipeline.predict(
            context_tensor,
            prediction_length=prediction_length
        )

    median_forecast = (
        prediction[
            0,
            4,
            :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return median_forecast