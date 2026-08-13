"""
Feature-based machine-learning forecasting models.
"""

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)


def create_hist_gradient_boosting(
    random_state=42
):
    """
    Create the HistGradientBoosting forecasting model.
    """
    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=20,
        l2_regularization=0.1,
        random_state=random_state,
    )


def train_feature_model(
    X_train,
    y_train,
    random_state=42
):
    """
    Train a HistGradientBoosting model.
    """
    model = create_hist_gradient_boosting(
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def predict_feature_model(
    model,
    X_test
):
    """Generate feature-model predictions."""
    return model.predict(X_test)