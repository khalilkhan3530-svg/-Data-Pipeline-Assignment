# Reproducible Data Pipelines, Tabular ML & Systems Exploration

## Project Overview

This project analyzes the UCI Online Retail dataset using Python, Pandas, NumPy, Matplotlib, SQLite, and Scikit-learn.

The project contains two main parts:

- **Part A:** Data inspection, cleaning, validation, analysis, visualization, and SQL.
- **Part B:** Machine learning pipeline for predicting next-day product sales.

## Dataset

- Dataset: UCI Online Retail
- Source: https://archive.ics.uci.edu/dataset/352/online-retail
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Original dataset: 541,909 transaction records.
- Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

The project uses different dataset samples for different tasks:

- Part A uses the first 8,000 rows.
- Part B uses the first 100,000 chronological transaction rows.

The original raw dataset is kept unchanged.

## Part A — Data Quality & Insights Pipeline

### Data Cleaning

- Removed duplicate rows.
- Filled missing product descriptions using StockCode.
- Removed rows where descriptions remained missing.
- Converted InvoiceDate into datetime format.
- Preserved negative quantities because they can represent returns or cancellations.
- Preserved zero prices for further investigation.

### Data Analysis

Used Pandas operations:

- groupby()
- agg()
- pivot_table()
- value_counts()

Used NumPy for vectorized operations.

### Visualizations

Created three visualizations:

1. Top 10 Countries by Number of Transactions
2. Top 10 Products by Net Quantity
3. Daily Net Sales

### SQL Analysis

Stored cleaned data in SQLite and performed a country-wise aggregation query.

### Automated Testing

Used pytest to test cleaning and validation functions.

## Part B — Machine Learning Pipeline

### Prediction Problem

The goal is to predict the next day's sales quantity for a product using its previous sales information.

### Feature Engineering

Created features including:

- StockCode
- sales_today
- sales_1_day_ago
- sales_2_days_ago
- sales_7_days_ago
- day_of_week

Target: Next-day product sales quantity.

### Data Splitting

Used chronological train-test splitting to avoid using future information to predict the past.

- Training samples: 610
- Testing samples: 160

### Models

Compared three regression models:

1. Dummy Regressor (Baseline)
2. Random Forest Regressor
3. Ridge Regression

Used Scikit-learn Pipeline and ColumnTransformer for preprocessing.

### Model Evaluation

The baseline model achieved the best test performance among the three candidates.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Dummy Regressor | 186.310 | 263.029 | -0.217 |
| Random Forest | 270.698 | 1193.653 | -24.057 |
| Ridge Regression | 353.528 | 541.159 | -4.150 |

The results show that the baseline performed better than the candidate models on this test sample.

### Error Analysis

Performed:

- Product-wise error analysis.
- Zero-sales vs positive-sales error analysis.
- Residual analysis.

### Saved Model

The selected fitted pipeline is saved at:

models/final_sales_pipeline.joblib

The pipeline was also tested on unseen product inputs, including an unknown product code.

### Model Card

Model details, intended use, evaluation, limitations, and risks are documented in:

model_card.md

## Project Outputs

- Cleaned dataset: data/processed/cleaned_online_retail.csv
- SQLite database: data/processed/online_retail.db
- Country analysis: data/processed/country_analysis.csv
- Model comparison: data/processed/model_comparison.csv
- Cross-validation results: data/processed/cross_validation_results.csv
- Product error analysis: data/processed/product_error_analysis.csv
- Sales type error analysis: data/processed/sales_type_error_analysis.csv
- Figures: outputs/figures/
- Quality metrics: outputs/metrics.json
- Trained pipeline: models/final_sales_pipeline.joblib

## Limitations

- Part A uses only the first 8,000 rows.
- Part B uses a limited sample and selected top products.
- The machine learning dataset contains a relatively small number of training and testing examples.
- The baseline model performed better than the candidate models.
- Negative R² scores indicate that the models do not yet provide strong predictive performance.
- The model is for educational experimentation and is not intended for production business decisions.

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset Download

To download and extract the original dataset:

```bash
python src/download_data.py
```

The dataset will be stored in:

data/raw/Online Retail.xlsx

## Run the Project

Run Part A:

```bash
python src/inspect_data.py
```

Run Part B:

```bash
python src/train_model.py
```

Run automated tests:

```bash
python -m pytest -v
```

## Author

Muhammad Khalil

## License

Dataset: CC BY 4.0 — UCI Online Retail Dataset.