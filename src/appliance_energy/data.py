"""
Data loading and preprocessing utilities.
"""

import pandas as pd


def load_energy_data(file_path):
    """
    Load appliance energy data.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Chronologically sorted dataset.
    """
    data = pd.read_csv(file_path)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"]
        )

        data = (
            data
            .set_index("date")
            .sort_index()
        )

    return data


def resample_hourly(data):
    """
    Resample 10-minute observations to hourly means.

    Parameters
    ----------
    data : pandas.DataFrame
        Original time-series dataset.

    Returns
    -------
    pandas.DataFrame
        Hourly dataset.
    """
    hourly_data = (
        data
        .resample("1h")
        .mean()
    )

    return hourly_data


def chronological_split(
    data,
    test_steps
):
    """
    Split time-series data chronologically.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Complete time-series data.
    test_steps : int
        Number of final observations reserved for testing.

    Returns
    -------
    tuple
        Training and test datasets.
    """
    train = data.iloc[:-test_steps].copy()
    test = data.iloc[-test_steps:].copy()

    return train, test