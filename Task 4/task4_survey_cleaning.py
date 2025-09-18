import os, re, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) CONFIG
# =========================
INPUT_PATH = r"C:/Users/ibadt/Downloads/archive2/kaggle_survey_2017_2021.csv"   # <- change this
OUT_DIR = "C:Users/ibadt/Desktop/DA TASKS/Task 4"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/insights", exist_ok=True)
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)

# =========================
# 1) LOAD
# =========================
# Works for .csv or .xlsx (requires openpyxl for xlsx)
if INPUT_PATH.lower().endswith(".xlsx"):
    df = pd.read_excel(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH, low_memory=False)

# Standardize column names
df.columns = (
    df.columns.str.strip()
              .str.replace("\n", " ", regex=False)
              .str.replace(r"\s+", " ", regex=True)
              .str.lower()
)

# =========================
# 2) CANONICAL COLUMN MAPPING (across years)
# (We’ll try to detect typical columns with fallbacks.)
# =========================
def first_match(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

cols = set(df.columns)

# Common variants across years (not exhaustive, but practical)
country_col = first_match(cols, ["country", "q3"])  # Q3 in many years
age_col     = first_match(cols, ["age", "q2"])
gender_col  = first_match(cols, ["gender", "q1"])
edu_col     = first_match(cols, ["education level", "highest level of formal education", "q4", "q6"])
comp_col    = first_match(cols, ["compensation", "salary", "q29", "q24", "q9", "current yearly compensation (approximate)"])
role_col    = first_match(cols, ["job title", "jobtitle", "q5"])
# Multi-select programming tools/questions change by year; this heuristic catches columns with "python", "r", "sql", etc.
tool_like_cols = [c for c in df.columns if re.search(r"(program|language|tool|python|r\b|sql|excel|tableau|power bi|spark|tensorflow|pytorch)", c)]

# Keep a copy of the raw for reference
raw_cols = {
    "country": country_col, "age": age_col, "gender": gender_col,
    "education": edu_col, "compensation": comp_col, "role": role_col
}

# =========================
# 3) BASIC CLEANING
# =========================
# Drop exact duplicates
df = df.drop_duplicates()

# Strip whitespace from object columns
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# Remove "Select all that apply" noise text in headers (common in Kaggle survey)
df = df.rename(columns=lambda c: re.sub(r"\(.*?select.*?apply.*?\)", "", c, flags=re.I).strip())

# =========================
# 4) CATEGORICAL NORMALIZATION
# =========================
# Normalize common categoricals (simple, safe mappings)
def normalize_gender(s):
    s = s.str.lower().str.replace(r"prefer.*not.*say", "prefer_not_say", regex=True)
    s = s.str.replace(r"male.*", "male", regex=True)
    s = s.str.replace(r"female.*", "female", regex=True)
    s = s.where(~s.isin(["nan","none",""]), other=np.nan)
    return s

def normalize_education(s):
    s = s.str.lower()
    s = s.replace({
        "primary/elementary school":"primary",
        "some college/university study without earning a bachelor’s degree":"some_college",
        "bachelor’s degree":"bachelors",
        "master’s degree":"masters",
        "doctoral degree":"phd",
        "professional degree":"professional",
        "i prefer not to answer":"prefer_not_say"
    })
    return s

if gender_col:
    df[gender_col] = normalize_gender(df[gender_col].astype(str))
if edu_col:
    df[edu_col] = normalize_education(df[edu_col].astype(str))

# =========================
# 5) COMPENSATION CLEANING
# =========================
def parse_comp(s):
    if pd.isna(s): return np.nan
    t = str(s).lower().strip()
    # Handle ranges like "$10,000-20,000" or "10,000-20,000"
    rng = re.findall(r"(\d[\d,\.]*)\s*[-–]\s*(\d[\d,\.]*)", t)
    if rng:
        a = float(rng[0][0].replace(",", ""))
        b = float(rng[0][1].replace(",", ""))
        return (a + b)/2
    # Single number like "50,000"
    num = re.findall(r"(\d[\d,\.]*)", t)
    if num:
        return float(num[0].replace(",", ""))
    return np.nan

if comp_col:
    df["compensation_usd"] = df[comp_col].apply(parse_comp)
else:
    df["compensation_usd"] = np.nan

# =========================
# 6) CREATE CLEAN SUBSET
# =========================
keep_cols = [c for c in [country_col, age_col, gender_col, edu_col, comp_col, role_col] if c]
keep_cols = list(dict.fromkeys(keep_cols))  # unique & preserve order
if "compensation_usd" not in keep_cols:
    keep_cols = keep_cols + ["compensation_usd"]

clean = df[keep_cols].copy()

# Convert age if numeric-like (some years are bins; keep as text then)
if age_col and pd.api.types.is_numeric_dtype(pd.to_numeric(clean[age_col], errors="coerce")):
    clean[age_col] = pd.to_numeric(clean[age_col], errors="coerce")

# Drop all-empty rows
clean = clean.dropna(how="all")

# =========================
# 7) LABEL ENCODING (simple)
# (Keep a human-readable copy; create an encoded one for ML)
# =========================
encoded = clean.copy()

label_maps = {}
for c in encoded.columns:
    if encoded[c].dtype == "object":
        cats = pd.Series(encoded[c].dropna().unique()).sort_values()
        mapping = {k:i for i,k in enumerate(cats, start=0)}
        label_maps[c] = mapping
        encoded[c] = encoded[c].map(mapping).astype("float")
# Note: compensation_usd stays numeric already.

# Save outputs
clean.to_csv(f"{OUT_DIR}/survey_clean.csv", index=False)
encoded.to_csv(f"{OUT_DIR}/survey_encoded.csv", index=False)

# =========================
# 8) INSIGHTS TABLES
# =========================
def save_table(df_, path, topn=None):
    out = df_.copy()
    if topn: out = out.head(topn)
    out.to_csv(path, index=True)

# Top Countries
if country_col:
    top_countries = clean[country_col].value_counts(dropna=False).head(20)
    save_table(top_countries, f"{OUT_DIR}/insights/top_countries.csv")

# Gender distribution
if gender_col:
    gender_dist = clean[gender_col].value_counts(dropna=False)
    save_table(gender_dist, f"{OUT_DIR}/insights/gender_distribution.csv")

# Education distribution
if edu_col:
    edu_dist = clean[edu_col].value_counts(dropna=False)
    save_table(edu_dist, f"{OUT_DIR}/insights/education_distribution.csv")

# Roles
if role_col:
    role_dist = clean[role_col].value_counts(dropna=False).head(20)
    save_table(role_dist, f"{OUT_DIR}/insights/top_roles.csv")

# Compensation summary
comp_summary = clean["compensation_usd"].describe().to_frame(name="compensation_usd")
comp_summary.to_csv(f"{OUT_DIR}/insights/compensation_summary.csv")

# =========================
# 9) CHARTS
# =========================
plt.figure(figsize=(8,5))
if country_col:
    clean[country_col].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Countries (Respondents)")
    plt.xlabel("Country"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/charts/top10_countries.png", dpi=150)
    plt.clf()

if gender_col:
    clean[gender_col].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/charts/gender_distribution.png", dpi=150)
    plt.clf()

if edu_col:
    clean[edu_col].value_counts().head(8).plot(kind="bar")
    plt.title("Top Education Levels")
    plt.xlabel("Education"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/charts/top_education.png", dpi=150)
    plt.clf()

if role_col:
    clean[role_col].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Roles")
    plt.xlabel("Role"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/charts/top_roles.png", dpi=150)
    plt.clf()

clean["compensation_usd"].dropna().plot(kind="hist", bins=40)
plt.title("Compensation (USD) — Distribution")
plt.xlabel("USD (approx.)"); plt.ylabel("Respondents")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/charts/compensation_hist.png", dpi=150)
plt.clf()

# =========================
# 10) TOP-5 INSIGHTS (AUTO)
# =========================
insights = []

if country_col:
    top_country = clean[country_col].value_counts().idxmax()
    insights.append(f"Most respondents are from {top_country}.")

if role_col:
    top_role = clean[role_col].value_counts().idxmax()
    insights.append(f"Most common role among respondents: {top_role}.")

if gender_col:
    g = clean[gender_col].value_counts(normalize=True)*100
    top_gender = g.idxmax()
    insights.append(f"Largest gender share: {top_gender} at {g[top_gender]:.1f}%.")

if edu_col:
    e = clean[edu_col].value_counts(normalize=True)*100
    top_edu = e.idxmax()
    insights.append(f"Most common education level: {top_edu} ({e[top_edu]:.1f}%).")

med = clean["compensation_usd"].median()
if pd.notna(med):
    insights.append(f"Median reported compensation ≈ ${med:,.0f}.")

# Ensure exactly 5 (pad if needed)
while len(insights) < 5:
    insights.append("Data varies by year; categories differ across surveys.")

with open(f"{OUT_DIR}/top5_insights.txt","w", encoding="utf-8") as f:
    f.write("Top 5 Insights\n")
    f.write("----------------\n")
    f.write("\n".join(f"- {i}" for i in insights[:5]))

print("Done ✅")
print(f"Outputs saved in: {os.path.abspath(OUT_DIR)}")
print("Detected columns:", raw_cols)
