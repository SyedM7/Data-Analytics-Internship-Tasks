# task9_olist_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Step 1: Load datasets
# -----------------------------
# Make sure these CSVs are in the same folder as the script
files = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv"
}

data = {}
for name, path in files.items():
    if os.path.exists(path):
        data[name] = pd.read_csv(path)
        print(f"Loaded {name}: {data[name].shape}")
    else:
        print(f"⚠️ File not found: {path}")

orders = data["orders"]
items = data["items"]
products = data["products"]
customers = data["customers"]
payments = data["payments"]
reviews = data["reviews"]

# -----------------------------
# Step 2: Data cleaning
# -----------------------------
# Convert date columns
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for col in date_cols:
    if col in orders:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Merge datasets
order_items = items.merge(products, on="product_id", how="left")
order_items = order_items.merge(orders, on="order_id", how="left")
order_items = order_items.merge(customers, on="customer_id", how="left")
order_items = order_items.merge(payments, on="order_id", how="left")
order_items = order_items.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

# Add revenue column
order_items["revenue"] = order_items["price"] + order_items["freight_value"]

# -----------------------------
# Step 3: Exploratory analysis
# -----------------------------
# Revenue trend over time
monthly_sales = order_items.groupby(order_items["order_purchase_timestamp"].dt.to_period("M")).agg(
    revenue=("revenue", "sum"),
    orders=("order_id", "nunique")
).reset_index()
monthly_sales["order_purchase_timestamp"] = monthly_sales["order_purchase_timestamp"].astype(str)

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_sales, x="order_purchase_timestamp", y="revenue", marker="o")
plt.xticks(rotation=45)
plt.title("Monthly Revenue Trend")
plt.tight_layout()
plt.savefig("monthly_revenue.png")
plt.close()

# Top product categories
cat_sales = order_items.groupby("product_category_name")["revenue"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=cat_sales.values, y=cat_sales.index)
plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig("top_categories.png")
plt.close()

# Payment types
pay_type = payments["payment_type"].value_counts()

plt.figure(figsize=(6, 6))
pay_type.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Payment Type Distribution")
plt.tight_layout()
plt.savefig("payment_types.png")
plt.close()

# Review score distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="review_score", data=reviews, palette="viridis")
plt.title("Customer Review Score Distribution")
plt.tight_layout()
plt.savefig("review_scores.png")
plt.close()

# Revenue by customer state
state_sales = order_items.groupby("customer_state")["revenue"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=state_sales.index, y=state_sales.values)
plt.title("Top 10 States by Revenue")
plt.ylabel("Revenue")
plt.xlabel("State")
plt.tight_layout()
plt.savefig("revenue_by_state.png")
plt.close()

# -----------------------------
# Step 4: Print key insights
# -----------------------------
print("\n📊 Business Insights:")
print(f"- Total Revenue: R$ {order_items['revenue'].sum():,.2f}")
print(f"- Total Orders: {orders['order_id'].nunique()}")
print(f"- Total Customers: {customers['customer_id'].nunique()}")

print("\nTop 5 Categories by Revenue:")
print(cat_sales.head())

print("\nTop 5 States by Revenue:")
print(state_sales.head())

avg_delivery = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days.mean()
print(f"\n- Average Delivery Time: {avg_delivery:.1f} days")

avg_review = reviews["review_score"].mean()
print(f"- Average Review Score: {avg_review:.2f}/5")

print("\n✅ Charts saved: monthly_revenue.png, top_categories.png, payment_types.png, review_scores.png, revenue_by_state.png")
