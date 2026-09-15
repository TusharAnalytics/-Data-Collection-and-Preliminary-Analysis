# Titanic Data Cleaning and Preprocessing Project
import pandas as pd
import numpy as np

# 1. Collect / load public dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
df = pd.read_csv(url)

# 2. Initial exploration
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))

# 3. Standardize categorical text
for col in ["Sex", "Embarked"]:
    df[col] = df[col].astype("string").str.strip().str.lower()

# 4. Handle missing Embarked values using the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 5. Handle missing Age using group-wise medians
group_medians = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(group_medians)
df["Age"] = df["Age"].fillna(df["Age"].median())

# 6. Remove high-missingness Cabin field for the baseline analysis
df = df.drop(columns=["Cabin"])

# 7. Detect Fare outliers with IQR
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["Fare_Outlier"] = ((df["Fare"] < lower) | (df["Fare"] > upper)).astype(int)

# 8. Reduce Fare skew without deleting valid passengers
df["Fare_Log"] = np.log1p(df["Fare"])

# 9. Encode categorical variables for downstream ML
df["Sex_Encoded"] = df["Sex"].map({"female": 0, "male": 1}).astype(int)
df["Embarked_Code"] = df["Embarked"].map({"c": 0, "q": 1, "s": 2}).astype(int)

# 10. Validation
print("Missing values after cleaning:")
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# 11. Save the processed dataset
df.to_csv("titanic_cleaned_preprocessed.csv", index=False)
print("Saved: titanic_cleaned_preprocessed.csv")
