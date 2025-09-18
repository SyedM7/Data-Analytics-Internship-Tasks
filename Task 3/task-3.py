# -----------------------------
# Customer Analytics with RFM
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
# Replace this with your own file path
df = pd.read_excel(
    "C:/Users/ibadt/Downloads/online+retail/Online Retail.xlsx",
    parse_dates=["InvoiceDate"]   # Excel date column
)

print(df.head())
print(df.info())

# 2. Data Cleaning
# Drop duplicates
df.drop_duplicates(inplace=True)

# Create Amount column
df["Amount"] = df["Quantity"] * df["UnitPrice"]

# Now drop rows with missing values
df = df.dropna(subset=["CustomerID", "InvoiceDate", "Amount"])

# Ensure correct datatypes
df["CustomerID"] = df["CustomerID"].astype(str)

# 3. Feature Engineering: RFM
# Define snapshot date (latest transaction + 1 day)
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,   # Recency
    "InvoiceDate": "count",                                   # Frequency
    "Amount": "sum"                                           # Monetary
})

rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceDate": "Frequency",
    "Amount": "Monetary"
}, inplace=True)

# Fix column overwrite bug (need to recalc Frequency separately)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceDate", "count"),
    Monetary=("Amount", "sum")
)

print("\nRFM table:")
print(rfm.head())

# 4. Segmentation with Quartiles
rfm["R_quartile"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F_quartile"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4])
rfm["M_quartile"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

# Combine into RFM Score
rfm["RFMScore"] = (
    rfm["R_quartile"].astype(str) +
    rfm["F_quartile"].astype(str) +
    rfm["M_quartile"].astype(str)
)

print("\nSegmented RFM table:")
print(rfm.head())

# 5. Visualization
plt.figure(figsize=(8,5))
sns.histplot(rfm["Recency"], bins=30, kde=True, color="blue")
plt.title("Recency Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(rfm["Frequency"], bins=30, kde=False, color="green")
plt.title("Frequency Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(rfm["Monetary"], bins=30, kde=True, color="orange")
plt.title("Monetary Distribution")
plt.show()

# Heatmap for correlation
plt.figure(figsize=(6,4))
sns.heatmap(rfm[["Recency","Frequency","Monetary"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("RFM Correlation Heatmap")
plt.show()
