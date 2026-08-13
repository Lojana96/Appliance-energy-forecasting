"""
Reusable forecasting visualisations.
"""

import matplotlib.pyplot as plt


def plot_forecast(
    actual,
    predicted,
    model_name,
    save_path=None
):
    """
    Plot actual and predicted observations.
    """
    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    actual.plot(
        ax=ax,
        label="Actual",
        marker="o",
        linewidth=2
    )

    predicted.plot(
        ax=ax,
        label=model_name,
        linewidth=2
    )

    ax.set_title(
        f"{model_name} — Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Appliance Energy Use"
    )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax


def plot_rmse_comparison(
    results,
    save_path=None
):
    """
    Plot model RMSE comparison.
    """
    plot_data = (
        results
        .sort_values(
            "RMSE",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        plot_data["Model"],
        plot_data["RMSE"]
    )

    ax.set_xlabel("RMSE")
    ax.set_ylabel(
        "Forecasting Model"
    )

    ax.set_title(
        "Forecasting Model Comparison"
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax