# Reproducible Data Quality & Insights Pipeline

## Project Overview
This project analyzes the Online Retail dataset using Python, Pandas, NumPy, Matplotlib, and SQLite.

The purpose is to inspect, clean, validate, and analyze retail transaction data.

## Dataset
- Dataset: UCI Online Retail
- Original dataset contains retail transaction records.
- This project uses the first 8,000 rows.
- Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

## Data Cleaning
- Removed duplicate rows.
- Filled missing product descriptions using StockCode.
- Removed rows where descriptions remained missing.
- Converted InvoiceDate into datetime format.
- Preserved negative quantities because they can represent returns or cancellations.
- Preserved zero prices for further investigation.

## Data Analysis
Used Pandas operations:
- groupby()
- agg()
- pivot_table()
- value_counts()

Used NumPy for vectorized operations.

## Visualizations
Created three visualizations:
1. Top 10 Countries by Number of Transactions
2. Top 10 Products by Net Quantity
3. Daily Net Sales

## SQL Analysis
Stored cleaned data in SQLite and performed a country-wise aggregation query.

## Automated Testing
Used pytest to test cleaning and validation functions.

Current test result: 10 passed.

## Outputs
- Cleaned dataset: data/processed/cleaned_online_retail.csv
- SQLite database: data/processed/online_retail.db
- SQL analysis: data/processed/country_analysis.csv
- Graphs: outputs/figures/
- Quality metrics: outputs/metrics.json

## Limitations
- Only the first 8,000 rows were used.
- The selected sample covers approximately five days of transactions.
- CustomerID contains missing values.
- Some zero prices and negative quantities were preserved because they may have valid business meanings.

## How to Run
Install dependencies:

```bash
python -m pip install pandas numpy matplotlib pytest
```

Run the pipeline:

```bash
python src/inspect_data.py
```

Run tests:

```bash
python -m pytest -v
```