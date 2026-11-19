import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json

# Load saved 10,000 rows CSV file
df = pd.read_csv("data/raw/Online_Retail_8000.csv")

# 1. Dataset shape
print("Dataset Shape:")
print(df.shape)

# 2. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 3. Data types
print("\nData Types:")
print(df.dtypes)

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Basic statistics
print("\nBasic Statistics:")
print(df.describe())


# 6. Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())


# 7. Check negative quantities
print("\nNegative Quantities:")
print((df["Quantity"] < 0).sum())


# 8. Check zero prices
print("\nZero Unit Prices:")
print((df["UnitPrice"] == 0).sum())

# 9. Convert date for inspection
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nDate Range:")
print("Start:", df["InvoiceDate"].min())
print("End:", df["InvoiceDate"].max())



# Check negative quantity transactions
negative_data = df[df["Quantity"] < 0]

print("\nNegative Quantity Records:")
print(negative_data.head(10))

print("\nInvoice Numbers of Negative Records:")
print(negative_data["InvoiceNo"].head(10))

# Country-wise transaction count
print("\nTransactions by Country:")
print(df["Country"].value_counts().head())


# Total quantity sold by country
country_sales = df.groupby("Country")["Quantity"].sum()

print("\nTotal Quantity by Country:")
print(country_sales.sort_values(ascending=False).head(10))



# Country-wise summary using Pivot Table

country_summary = pd.pivot_table(
    df,
    index="Country",
    values=["Quantity", "UnitPrice"],
    aggfunc={
        "Quantity": "sum",
        "UnitPrice": "mean"
    }
)

print("\nCountry-wise Summary:")
print(country_summary.sort_values(by="Quantity", ascending=False).head(10))




# Convert UnitPrice column into NumPy array
prices = df["UnitPrice"].to_numpy()

# Increase every price by 10%
increased_prices = prices * 1.10

print("\nOriginal Prices:")
print(prices[:10])

print("\nPrices After 10% Increase:")
print(increased_prices[:10])


# Convert Quantity column into NumPy array
quantities = df["Quantity"].to_numpy()

# Identify negative quantities
negative_mask = quantities < 0

# Count negative quantities
negative_count = np.sum(negative_mask)

print("\nTotal Negative Quantities:")
print(negative_count)


# Convert InvoiceDate into datetime format

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nUpdated Data Types:")
print(df.dtypes)

print("\nFirst 5 Invoice Dates:")
print(df["InvoiceDate"].head())



# Calculate missing CustomerID percentage

missing_percentage = df["CustomerID"].isnull().mean() * 100

print("\nMissing CustomerID Percentage:")
print(round(missing_percentage, 2), "%")


# Create CustomerID status column

df["CustomerID_Status"] = np.where(
    df["CustomerID"].isnull(),
    "Missing",
    "Available"
)

print("\nCustomerID Status:")
print(df["CustomerID_Status"].value_counts())

# Inspect duplicate records

duplicate_data = df[df.duplicated(keep=False)]

print("\nDuplicate Records:")
print(duplicate_data.head(10))

print("\nTotal Duplicate Records:")
print(len(duplicate_data))


# Create a copy for cleaning
cleaned_df = df.copy()

# Remove completely duplicate rows
cleaned_df = cleaned_df.drop_duplicates()

# Compare rows before and after cleaning
print("\nRows Before Cleaning:")
print(len(df))

print("\nRows After Removing Duplicates:")
print(len(cleaned_df))

print("\nDuplicates Remaining:")
print(cleaned_df.duplicated().sum())


# Find records with missing descriptions

missing_description = cleaned_df[
    cleaned_df["Description"].isnull()
]

print("\nRecords with Missing Description:")
print(missing_description.head(10))

print("\nTotal Missing Descriptions:")
print(len(missing_description))



# Find available descriptions for each StockCode

description_lookup = (
    cleaned_df.dropna(subset=["Description"])
    .drop_duplicates(subset=["StockCode"])
    .set_index("StockCode")["Description"]
)

# Fill missing descriptions using StockCode
cleaned_df["Description"] = (
    cleaned_df["Description"]
    .fillna(cleaned_df["StockCode"].map(description_lookup))
)

# Check remaining missing descriptions
print("\nMissing Descriptions After Filling:")
print(cleaned_df["Description"].isnull().sum())


cleaned_df = cleaned_df.dropna(subset=["Description"])

print("Rows after removing missing descriptions:", len(cleaned_df))
print("Missing descriptions remaining:", cleaned_df["Description"].isnull().sum())


print("CustomerID missing before cleaning:", cleaned_df["CustomerID"].isnull().sum())

print("CustomerID missing percentage:",
      round(cleaned_df["CustomerID"].isnull().mean() * 100, 2), "%")


zero_price = cleaned_df[cleaned_df["UnitPrice"] == 0]

print("Zero UnitPrice records:", len(zero_price))
print(zero_price[["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"]].head())


negative_quantity = cleaned_df[cleaned_df["Quantity"] < 0]

print("Negative Quantity records:", len(negative_quantity))
print(negative_quantity[["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"]].head())


print("Highest 10 Quantities:")
print(cleaned_df["Quantity"].nlargest(10))

print("\nHighest 10 UnitPrices:")
print(cleaned_df["UnitPrice"].nlargest(10))


high_quantity = cleaned_df.nlargest(10, "Quantity")

print(high_quantity[["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"]])


Q1 = cleaned_df["Quantity"].quantile(0.25)
Q3 = cleaned_df["Quantity"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


outliers = cleaned_df[
    (cleaned_df["Quantity"] < lower_limit) |
    (cleaned_df["Quantity"] > upper_limit)
]

print("Total Quantity Outliers:", len(outliers))
print(outliers[["Description", "Quantity"]].head(10))



Q1_price = cleaned_df["UnitPrice"].quantile(0.25)
Q3_price = cleaned_df["UnitPrice"].quantile(0.75)

IQR_price = Q3_price - Q1_price

upper_price_limit = Q3_price + 1.5 * IQR_price

price_outliers = cleaned_df[
    cleaned_df["UnitPrice"] > upper_price_limit
]

print("Upper Price Limit:", upper_price_limit)
print("Total Price Outliers:", len(price_outliers))


print(price_outliers[["Description", "Quantity", "UnitPrice"]].nlargest(10, "UnitPrice"))

def remove_duplicates(data):
    """Remove duplicate rows from the dataset."""
    return data.drop_duplicates()

test_df = remove_duplicates(df)

print("Original rows:", len(df))
print("Rows after removing duplicates:", len(test_df))
print("Remaining duplicates:", test_df.duplicated().sum())


def remove_missing_descriptions(data):
    """Remove rows where Description is missing."""
    return data.dropna(subset=["Description"])

test_df = remove_missing_descriptions(cleaned_df)

print("Rows before:", len(cleaned_df))
print("Rows after:", len(test_df))
print("Missing descriptions:", test_df["Description"].isnull().sum())


def convert_date(data):
    """Convert InvoiceDate column into datetime format."""
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    return data


test_df = convert_date(cleaned_df.copy())

print("InvoiceDate data type:", test_df["InvoiceDate"].dtype)
print("First 5 dates:")
print(test_df["InvoiceDate"].head())


def validate_quantity(data):
    """Check whether Quantity column contains numeric values."""
    return pd.api.types.is_numeric_dtype(data["Quantity"])

result = validate_quantity(cleaned_df)

print("Quantity validation:", result)


def validate_missing_values(data):
    """Check whether Description column has missing values."""
    return data["Description"].isnull().sum() == 0

result = validate_missing_values(cleaned_df)

print("Missing values validation:", result)


def validate_duplicates(data):
    """Check whether duplicate rows exist in the dataset."""
    return data.duplicated().sum() == 0

result = validate_duplicates(cleaned_df)

print("Duplicate validation:", result)




def fill_missing_descriptions(data):
    """Fill missing descriptions using matching StockCode."""
    description_lookup = (
        data.dropna(subset=["Description"])
        .drop_duplicates(subset=["StockCode"])
        .set_index("StockCode")["Description"]
    )

    data["Description"] = data["Description"].fillna(
        data["StockCode"].map(description_lookup)
    )

    return data

pipeline_df = remove_duplicates(df.copy())
pipeline_df = convert_date(pipeline_df)
pipeline_df = fill_missing_descriptions(pipeline_df)
pipeline_df = remove_missing_descriptions(pipeline_df)

print("Final rows:", len(pipeline_df))
print("Remaining duplicates:", pipeline_df.duplicated().sum())
print("Missing descriptions:", pipeline_df["Description"].isnull().sum())


# Create SQLite Database

connection = sqlite3.connect(
    "data/processed/online_retail.db"
)

cleaned_df.to_sql(
    "retail_data",
    connection,
    if_exists="replace",
    index=False
)

print("Data saved into SQLite database!")


# SQL Query: Records by Country

query = """
SELECT
    Country,
    COUNT(*) AS Total_Records,
    SUM(Quantity) AS Total_Quantity
FROM retail_data
GROUP BY Country
ORDER BY Total_Records DESC
LIMIT 10;
"""

sql_result = pd.read_sql_query(query, connection)

print(sql_result)

connection.close()

sql_result.to_csv(
    "data/processed/country_analysis.csv",
    index=False
)
cleaned_df.to_csv(
    "data/processed/cleaned_online_retail.csv",
    index=False
)



# Data Quality Metrics

quality_metrics = {
    "total_rows": len(cleaned_df),
    "total_columns": len(cleaned_df.columns),
    "missing_descriptions": int(cleaned_df["Description"].isnull().sum()),
    "missing_customer_ids": int(cleaned_df["CustomerID"].isnull().sum()),
    "duplicate_rows": int(cleaned_df.duplicated().sum()),
    "negative_quantities": int((cleaned_df["Quantity"] < 0).sum()),
    "zero_prices": int((cleaned_df["UnitPrice"] == 0).sum())
}

with open("outputs/metrics.json", "w") as file:
    json.dump(quality_metrics, file, indent=4)

print("Quality metrics saved!")

# # Top 10 Countries by Number of Transactions

# top_countries = (
#     cleaned_df["Country"]
#     .value_counts()
#     .head(10)
# )

# plt.figure(figsize=(10, 5))

# top_countries.plot(kind="bar")

# plt.title("Top 10 Countries by Number of Transactions")
# plt.xlabel("Country")
# plt.ylabel("Number of Transactions")

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("outputs/figures/top_countries.png")

# plt.show()



# # Top 10 Products by Net Quantity

# top_products = (
#     cleaned_df.groupby("Description")["Quantity"]
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
# )

# plt.figure(figsize=(10, 6))

# top_products.plot(kind="barh")

# plt.title("Top 10 Products by Net Quantity")
# plt.xlabel("Net Quantity")
# plt.ylabel("Product")

# plt.tight_layout()
# plt.savefig("outputs/figures/top_products.png")
# plt.show()


# # Daily Net Sales

# daily_df = cleaned_df.copy()

# daily_df["InvoiceDay"] = daily_df["InvoiceDate"].dt.date

# daily_df["Sales"] = (
#     daily_df["Quantity"] * daily_df["UnitPrice"]
# )

# daily_sales = (
#     daily_df.groupby("InvoiceDay")["Sales"]
#     .sum()
# )

# plt.figure(figsize=(10, 5))

# daily_sales.plot(kind="line", marker="o")

# plt.title("Daily Net Sales")
# plt.xlabel("Date")
# plt.ylabel("Net Sales (£)")

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("outputs/figures/daily_net_sales.png")

# plt.show()