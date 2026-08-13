# Intelligent Appliance Energy Forecasting Using Statistical, Machine Learning, and Foundation Models

## Project Overview

This project investigates short-term household appliance energy forecasting using multiple forecasting paradigms, ranging from traditional statistical models to modern foundation models.

The objective is to compare forecasting performance across benchmark models, SARIMAX, feature-based machine learning, and Chronos-Bolt foundation models using the UCI Appliances Energy Prediction dataset.

The project follows a complete data science workflow including:

- Data preparation
- Exploratory data analysis
- Benchmark forecasting
- SARIMAX modelling
- Feature engineering
- Feature-based machine learning
- Foundation model forecasting
- Model comparison and evaluation

---

## Dataset

**Dataset:**
UCI Machine Learning Repository – Appliances Energy Prediction Dataset

https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

The dataset contains hourly household appliance energy consumption together with indoor environmental sensor measurements and outdoor weather variables.

---

## Project Structure

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   ├── models/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── pipeline.py
│   └── plotting.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   └── metrics/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Forecasting Models

### Benchmark Models

- Mean
- Naïve
- Drift
- Daily Seasonal Naïve
- Weekly Seasonal Naïve

### Statistical Model

- SARIMAX

### Feature-Based Machine Learning

- HistGradientBoostingRegressor

### Foundation Model

- Chronos-Bolt-Tiny

---

## Evaluation Metrics

The following metrics were used:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)
- Forecast Bias

---

## Main Results

| Model | RMSE |
|-------|------|
| SARIMAX | 24.158 |
| HistGradientBoosting | 29.865 |
| Chronos-Bolt-Tiny | 42.226 |
| Weekly Seasonal Naïve | 48.808 |

The SARIMAX model achieved the best forecasting performance over the common 24-hour evaluation period.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Chronos-Bolt
- Jupyter Notebook

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/appliance-energy-forecasting.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Run notebooks sequentially:

1. Data Preparation
2. Exploratory Analysis
3. Benchmark Models
4. SARIMAX
5. Feature-Based Models
6. Foundation Model
7. Model Comparison

---

## Author

Lojana Jegatheeswaran

MSc Data Science

University of Hertfordshire