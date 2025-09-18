import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


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

df["Amount"] = df["Quantity"] * df["UnitPrice"]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
    "InvoiceNo": "count",                                     # Frequency
    "Amount": "sum"                                           # Monetary
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)

print(rfm.head())
print(rfm.sort_values("RFM_Score", ascending=False).head(10))  # Top customers

sns.histplot(rfm["Recency"], bins=30, kde=True)
plt.title("Recency Distribution")
plt.show()

sns.histplot(rfm["Frequency"], bins=30, kde=True)
plt.title("Frequency Distribution")
plt.show()

sns.histplot(rfm["Monetary"], bins=30, kde=True)
plt.title("Monetary Distribution")
plt.show()
