"""
Project configuration and shared constants.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"

TARGET_COLUMN = "Appliances"

DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

FORECAST_HORIZON = 24
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24

RANDOM_STATE = 42


def create_project_directories():
    """Create required project output directories."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURE_DIR,
        FORECAST_DIR,
        METRICS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )