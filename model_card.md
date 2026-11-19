# Model Card: Online Retail Sales Prediction

## 1. Project Overview
This project predicts next-day product sales quantity using historical daily sales data.

## 2. Dataset
- Dataset: UCI Online Retail Dataset
- Rows selected: 100,000
- Products selected: Top 10 products
- Data type: Retail transaction data

## 3. Prediction Target
The target is the product's sales quantity on the next calendar day.

## 4. Features
- StockCode
- sales_today
- sales_1_day_ago
- sales_2_days_ago
- sales_7_days_ago
- day_of_week

## 5. Data Preparation
- Positive sales records were selected.
- Daily product sales were calculated.
- Missing sales days were filled with zero.
- Data was split chronologically into training and testing sets.
- Numerical features were imputed and scaled.
- Product codes were encoded using OneHotEncoder.

## 6. Models Evaluated
- Dummy Regressor (Baseline)
- Random Forest Regressor
- Ridge Regression

## 7. Model Evaluation
The Dummy Regressor performed better than both candidate models on the test set and cross-validation.

Test results:
- MAE: 186.31
- RMSE: 263.03
- R²: -0.217

## 8. Selected Model
Dummy Regressor with mean strategy was selected because it achieved the best evaluation results among the tested models.

## 9. Limitations
- Only 100,000 transaction rows were selected.
- Only the top 10 products were used.
- The dataset covers a limited date range.
- Sales data contains large variations and unusual high-volume transactions.
- The selected baseline predicts the same average value for every input.
- The model is not recommended for production use.

## 10. Future Improvements
- Use more historical data.
- Evaluate additional machine learning models.
- Handle extreme sales values carefully.
- Improve product-specific demand prediction.