#Name : Muhammad khalil
#ID : TRN-2026-000376
#Date : 19-11-2026



import pandas as pd

# Load selected dataset
df = pd.read_csv("data/processed/retail_ml_data.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset Shape:", df.shape)
print("\nDate Range:")
print(df["InvoiceDate"].min())
print(df["InvoiceDate"].max())




# Extract date only
df["Date"] = df["InvoiceDate"].dt.date

# Calculate daily product sales
daily_sales = df.groupby(
    ["Date", "StockCode"]
)["Quantity"].sum().reset_index()

print("\nDaily Sales:")
print(daily_sales.head(10))

print("\nDaily Sales Shape:", daily_sales.shape)




# Keep only positive sales quantities
sales_data = df[df["Quantity"] > 0].copy()

# Extract date only
sales_data["Date"] = sales_data["InvoiceDate"].dt.date

# Calculate daily product sales
daily_sales = sales_data.groupby(
    ["Date", "StockCode"]
)["Quantity"].sum().reset_index()

print("\nDaily Sales After Removing Returns:")
print(daily_sales.head(10))

print("\nUpdated Daily Sales Shape:", daily_sales.shape)

print("\nNegative Records:",
      (daily_sales["Quantity"] < 0).sum())




# Find top 10 selling products
top_products = (
    daily_sales.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products by Total Sales:")
print(top_products)




# Select only top 10 products
top_product_codes = top_products.index

top_sales = daily_sales[
    daily_sales["StockCode"].isin(top_product_codes)
].copy()

# Create all dates in the dataset
all_dates = pd.date_range(
    start=daily_sales["Date"].min(),
    end=daily_sales["Date"].max(),
    freq="D"
).date

# Create every product-date combination
full_index = pd.MultiIndex.from_product(
    [top_product_codes, all_dates],
    names=["StockCode", "Date"]
)

# Fill missing product-days with zero
daily_sales_complete = (
    top_sales.set_index(["StockCode", "Date"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

# Sort the data
daily_sales_complete = daily_sales_complete.sort_values(
    ["StockCode", "Date"]
)

print("\nComplete Daily Sales:")
print(daily_sales_complete.head(15))

print("\nComplete Dataset Shape:", daily_sales_complete.shape)




# Create ML dataset
ml_data = daily_sales_complete.copy()

# Group sales by product
grouped = ml_data.groupby("StockCode")["Quantity"]

# Create sales history features
ml_data["sales_today"] = grouped.shift(0)
ml_data["sales_1_day_ago"] = grouped.shift(1)
ml_data["sales_2_days_ago"] = grouped.shift(2)
ml_data["sales_7_days_ago"] = grouped.shift(7)

# Next day sales as target
ml_data["target"] = grouped.shift(-1)

# Remove missing values
ml_data = ml_data.dropna()

print("\nML Dataset:")
print(ml_data.head(10))

print("\nML Dataset Shape:", ml_data.shape)




# Save prepared ML dataset
ml_data.to_csv(
    "data/processed/retail_ml_dataset.csv",
    index=False
)

print("\nML dataset saved successfully!")




# Define input features
feature_cols = [
    "StockCode",
    "sales_today",
    "sales_1_day_ago",
    "sales_2_days_ago",
    "sales_7_days_ago"
]

# Separate features and target
X = ml_data[feature_cols]
y = ml_data["target"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)




# Sort data by date
ml_data = ml_data.sort_values("Date")

# Get unique dates
unique_dates = sorted(ml_data["Date"].unique())

# Find 80% date boundary
split_index = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_index]

# Split training and testing data
train_data = ml_data[ml_data["Date"] < split_date]
test_data = ml_data[ml_data["Date"] >= split_date]

# Separate X and y
X_train = train_data[feature_cols]
y_train = train_data["target"]

X_test = test_data[feature_cols]
y_test = test_data["target"]

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Date Range:")
print(train_data["Date"].min(), "to", train_data["Date"].max())

print("\nTesting Date Range:")
print(test_data["Date"].min(), "to", test_data["Date"].max())





# Convert Date columns to datetime
ml_data["Date"] = pd.to_datetime(ml_data["Date"])
train_data["Date"] = pd.to_datetime(train_data["Date"])
test_data["Date"] = pd.to_datetime(test_data["Date"])

# Create day of week feature
train_data["day_of_week"] = train_data["Date"].dt.dayofweek
test_data["day_of_week"] = test_data["Date"].dt.dayofweek

# Update feature columns
feature_cols = [
    "StockCode",
    "sales_today",
    "sales_1_day_ago",
    "sales_2_days_ago",
    "sales_7_days_ago",
    "day_of_week"
]

# Separate features and target
X_train = train_data[feature_cols]
y_train = train_data["target"]

X_test = test_data[feature_cols]
y_test = test_data["target"]

print("\nUpdated Features:")
print(X_train.head())

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Step 15: Inspect Target Variable

print("\nTraining Target Summary:")
print(y_train.describe())

print("\nTesting Target Summary:")
print(y_test.describe())

print("\nMissing Values in Training Features:")
print(X_train.isnull().sum())

print("\nMissing Values in Target:")
print(y_train.isnull().sum())


# Step 16: Inspect Unusual Sales Values

print("\nHighest Training Targets:")

print(
    train_data[
        ["StockCode", "Date", "target"]
    ].sort_values(
        "target", ascending=False
    ).head(10)
)






# Step 17: Create Preprocessing Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Make copies of training and testing features
X_train = X_train.copy()
X_test = X_test.copy()

# Convert product codes into strings
X_train["StockCode"] = X_train["StockCode"].astype(str)
X_test["StockCode"] = X_test["StockCode"].astype(str)

# Numerical features
numeric_features = [
    "sales_today",
    "sales_1_day_ago",
    "sales_2_days_ago",
    "sales_7_days_ago",
    "day_of_week"
]

# Categorical features
categorical_features = ["StockCode"]

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both transformers
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed!")

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)


# Step 18: Train Baseline Model

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create baseline model
baseline_model = DummyRegressor(strategy="mean")

# Train model
baseline_model.fit(X_train_processed, y_train)

# Make predictions
baseline_predictions = baseline_model.predict(X_test_processed)

# Evaluate model
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

baseline_r2 = r2_score(y_test, baseline_predictions)

print("\nBaseline Model Results:")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2 Score:", baseline_r2)


# Step 19: Random Forest Model

from sklearn.ensemble import RandomForestRegressor

# Create model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train_processed, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test_processed)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# Step 20: Ridge Regression Model

from sklearn.linear_model import Ridge

# Create model
ridge_model = Ridge(alpha=1.0)

# Train model
ridge_model.fit(X_train_processed, y_train)

# Predictions
ridge_predictions = ridge_model.predict(X_test_processed)

# Evaluation
ridge_mae = mean_absolute_error(y_test, ridge_predictions)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, ridge_predictions)
)

ridge_r2 = r2_score(y_test, ridge_predictions)

print("\nRidge Regression Results:")
print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)
print("R2 Score:", ridge_r2)


# Step 22: Compare All Models

import pandas as pd

results = pd.DataFrame({
    "Model": [
        "Baseline",
        "Random Forest",
        "Ridge Regression"
    ],
    "MAE": [
        baseline_mae,
        rf_mae,
        ridge_mae
    ],
    "RMSE": [
        baseline_rmse,
        rf_rmse,
        ridge_rmse
    ],
    "R2 Score": [
        baseline_r2,
        rf_r2,
        ridge_r2
    ]
})

print("\nModel Comparison:")
print(results.round(3))

# Save comparison table
results.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison table saved successfully!")





# Step 23: Time Series Cross Validation

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

# Sort training data by date
train_data = train_data.sort_values("Date")

# Get unique training dates
date_values = pd.to_datetime(train_data["Date"]).to_numpy()
unique_dates = np.sort(np.unique(date_values))

# Create time-based splits by date
tscv = TimeSeriesSplit(n_splits=5)

cv_splits = []

for train_idx, val_idx in tscv.split(unique_dates):

    train_dates = unique_dates[train_idx]
    val_dates = unique_dates[val_idx]

    fold_train_idx = np.where(
        np.isin(date_values, train_dates)
    )[0]

    fold_val_idx = np.where(
        np.isin(date_values, val_dates)
    )[0]

    cv_splits.append((fold_train_idx, fold_val_idx))


# Create complete model pipelines
models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Ridge Regression": Ridge(alpha=1.0)
}

cv_results = []

for model_name, model in models.items():

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        cv=cv_splits,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        }
    )

    cv_results.append({
        "Model": model_name,
        "CV MAE": -scores["test_MAE"].mean(),
        "CV RMSE": -scores["test_RMSE"].mean(),
        "CV R2": scores["test_R2"].mean()
    })

cv_table = pd.DataFrame(cv_results)

print("\nCross Validation Results:")
print(cv_table.round(3))

cv_table.to_csv(
    "data/processed/cross_validation_results.csv",
    index=False
)

print("\nCross Validation Results Saved!")



# Step 24: Error Analysis

# Create error analysis dataset
error_analysis = pd.DataFrame({
    "StockCode": X_test["StockCode"].values,
    "Actual": y_test.values,
    "Predicted": rf_predictions
})

# Calculate prediction errors
error_analysis["Absolute_Error"] = abs(
    error_analysis["Actual"] - error_analysis["Predicted"]
)

# Calculate error for each product
product_errors = error_analysis.groupby("StockCode").agg(
    Total_Records=("Actual", "count"),
    Average_Actual=("Actual", "mean"),
    Average_Predicted=("Predicted", "mean"),
    MAE=("Absolute_Error", "mean")
).reset_index()

# Sort products by highest error
product_errors = product_errors.sort_values(
    "MAE", ascending=False
)

print("\nProduct-wise Error Analysis:")
print(product_errors.round(2))

# Show 10 largest individual prediction errors
print("\nTop 10 Prediction Errors:")

print(
    error_analysis.sort_values(
        "Absolute_Error", ascending=False
    ).head(10)
)

# Save error analysis
product_errors.to_csv(
    "data/processed/product_error_analysis.csv",
    index=False
)

print("\nError Analysis Saved!")



# Step 25: Zero vs Positive Sales Error Analysis

error_analysis["Sales_Type"] = np.where(
    error_analysis["Actual"] == 0,
    "Zero Sales",
    "Positive Sales"
)

sales_type_analysis = error_analysis.groupby("Sales_Type").agg(
    Total_Records=("Actual", "count"),
    Average_Actual=("Actual", "mean"),
    Average_Predicted=("Predicted", "mean"),
    MAE=("Absolute_Error", "mean")
).reset_index()

print("\nZero vs Positive Sales Analysis:")
print(sales_type_analysis.round(2))

sales_type_analysis.to_csv(
    "data/processed/sales_type_error_analysis.csv",
    index=False
)

print("\nSales Type Analysis Saved!")



# Step 26: Save Final Fitted Pipeline

import os
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Create complete pipeline
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DummyRegressor(strategy="mean"))
])

# Train final pipeline
final_pipeline.fit(X_train, y_train)

# Save pipeline
dump(final_pipeline, "models/final_sales_pipeline.joblib")

print("\nFinal Pipeline Saved Successfully!")







# Step 26: Unseen Data Inference

import pandas as pd
from joblib import load

# Load saved pipeline
loaded_pipeline = load("models/final_sales_pipeline.joblib")

# Create new unseen data
unseen_data = pd.DataFrame({
    "StockCode": ["21212", "NEW_PRODUCT"],
    "sales_today": [100, 25],
    "sales_1_day_ago": [80, 30],
    "sales_2_days_ago": [75, 20],
    "sales_7_days_ago": [60, 10],
    "day_of_week": [2, 3]
})

# Make predictions
predictions = loaded_pipeline.predict(unseen_data)

print("\nUnseen Data Predictions:")

for product, prediction in zip(unseen_data["StockCode"], predictions):
    print(f"Product: {product}, Predicted Next-Day Sales: {prediction:.2f}")




# Step 29: Residual Plot

import matplotlib.pyplot as plt
import os

residuals = y_test.values - rf_predictions

plt.figure(figsize=(8, 5))

plt.scatter(rf_predictions, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Random Forest Residual Plot")

os.makedirs("outputs/figures", exist_ok=True)

plt.savefig("outputs/figures/residual_plot.png")
plt.close()

print("Residual Plot Saved Successfully!")
