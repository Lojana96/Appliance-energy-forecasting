"""
Main reusable project pipeline.

Expensive hyperparameter searches are intentionally
excluded from automatic execution.
"""

from .config import (
    PROCESSED_DATA_DIR,
    TEST_STEPS,
    TARGET_COLUMN,
    create_project_directories,
)

from .data import (
    load_energy_data,
    chronological_split,
)

from .features import (
    build_feature_table,
)

from .models.feature_models import (
    train_feature_model,
)


def run_pipeline():
    """
    Run a lightweight demonstration pipeline.
    """
    create_project_directories()

    data_file = (
        PROCESSED_DATA_DIR /
        "appliance_energy_hourly.csv"
    )

    data = load_energy_data(
        data_file
    )

    feature_data = build_feature_table(
        data,
        target=TARGET_COLUMN
    )

    train, test = chronological_split(
        feature_data,
        TEST_STEPS
    )

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_std_24",
        "T1",
        "RH_1",
        "T_out",
        "RH_out",
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column in feature_data.columns
    ]

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_test = test[
        feature_columns
    ]

    model = train_feature_model(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    print(
        "Pipeline completed successfully."
    )

    print(
        "Predictions generated:",
        len(predictions)
    )

    return predictions


if __name__ == "__main__":
    run_pipeline()