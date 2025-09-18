import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset (adjust path if needed)
df = pd.read_csv(r"C:/Users/ibadt/Downloads/titanic/train.csv")



# First look at the data
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Convert Sex to numeric (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numeric
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Overall survival rate
print("Overall survival rate:", df['Survived'].mean())

# Survival by gender
print(df.groupby('Sex')['Survived'].mean())

# Survival by class
print(df.groupby('Pclass')['Survived'].mean())

# Survival by gender and class
print(df.groupby(['Pclass', 'Sex'])['Survived'].mean())


# Survival by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

# Survival by passenger class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Class")
plt.show()

# Age distribution by survival
sns.histplot(df[df['Survived']==1]['Age'], bins=30, kde=True, color="green", label="Survived")
sns.histplot(df[df['Survived']==0]['Age'], bins=30, kde=True, color="red", label="Not Survived")
plt.legend()
plt.title("Age Distribution by Survival")
plt.show()

# Only keep numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
