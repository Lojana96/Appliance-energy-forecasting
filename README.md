# Appliance Energy Consumption Forecasting Using Classical Time Series, Machine Learning and Foundation Models

## Overview

This project investigates short-term household appliance energy consumption forecasting using multiple forecasting approaches, ranging from simple benchmark models to advanced statistical, machine learning and foundation models.

The project follows a complete end-to-end data science workflow including:

- Data preprocessing
- Exploratory time-series analysis
- Benchmark forecasting
- SARIMAX modelling
- Feature-based machine learning
- Foundation model forecasting (Chronos-Bolt-Tiny)
- Model comparison and evaluation

The objective is to identify the most accurate and practical forecasting model for smart-home energy management.

---

## Dataset

**Source**

UCI Machine Learning Repository

**Dataset**

Appliance Energy Prediction Dataset

Original paper:

> Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house.

---

## Project Structure

```
Appliance-energy-forecasting/
│
├── data/
│   └── Raw and processed datasets
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_model.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_final_model_comparison.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── trained_models/
│
├── reports/
│   └── Final Report (PDF)
│
├── scripts/
│   └── Utility scripts
│
├── src/
│   └── appliance_energy/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── plotting.py
│       ├── pipeline.py
│       ├── __init__.py
│       └── models/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Forecasting Models

### Benchmark Models

- Mean
- Naïve
- Daily Seasonal Naïve
- Weekly Seasonal Naïve
- Drift

### Statistical Model

- SARIMAX

### Machine Learning Model

- HistGradientBoosting Regressor

### Foundation Model

- Chronos-Bolt-Tiny

---

## Evaluation Metrics

The forecasting models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)
- Forecast Bias

---

## Results

### 14-Day Rolling Evaluation

| Model | RMSE |
|-------|------:|
| HistGradientBoosting | **52.26** |
| SARIMAX | 65.14 |
| Mean Benchmark | 74.91 |
| Weekly Seasonal Naïve | 79.29 |
| Chronos-Bolt-Tiny* | 42.23 (24-hour evaluation) |

*Chronos was evaluated separately on a common 24-hour forecasting horizon.

### Final 24-Hour Comparison

| Model | RMSE |
|-------|------:|
| SARIMAX | **24.16** |
| HistGradientBoosting | 29.87 |
| Chronos-Bolt-Tiny | 42.23 |
| Weekly Seasonal Naïve | 48.81 |

---

## Key Findings

- Strong daily seasonality was identified during exploratory analysis.
- Weekly Seasonal Naïve was the strongest benchmark model.
- SARIMAX achieved the highest forecasting accuracy on the common 24-hour evaluation.
- HistGradientBoosting substantially improved forecasting performance through engineered temporal features.
- Chronos-Bolt-Tiny demonstrated promising zero-shot forecasting capability but did not outperform task-specific models.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn
- Chronos-Bolt
- Jupyter Notebook

---

## Author

**Lojana Jegatheeswaran**

MSc Data Science

University of Hertfordshire

2026

---

## License

This project is released under the MIT License.