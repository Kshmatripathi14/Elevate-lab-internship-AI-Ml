import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ------------------------------
# Load Dataset
# ------------------------------
# Replace with your dataset path if needed
df = pd.read_csv("data.csv")

# Explore dataset
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing values per column:\n", df.isnull().sum())

# ------------------------------
# Handle Missing Values
# ------------------------------
# Numerical: mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Categorical: mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after handling:\n", df.isnull().sum())

# ------------------------------
# Encode Categorical Features
# ------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:\n", df.head())

# ------------------------------
# Normalize / Standardize
# ------------------------------
num_cols = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()  # or MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nData after scaling:\n", df.head())

# ------------------------------
# Outlier Detection & Removal
# ------------------------------
plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, (len(num_cols) + 1) // 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in num_cols:
    df = remove_outliers_iqr(df, col)

print("\nData shape after outlier removal:", df.shape)

# ------------------------------
# Save Clean Dataset
# ------------------------------
df.to_csv("cleaned_data.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_data.csv'")



            OR  



# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ------------------------------
# 1. Import the dataset
# ------------------------------
# Replace 'data.csv' with your dataset path
df = pd.read_csv("data.csv")

# ------------------------------
# 2. Explore basic info
# ------------------------------
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing values per column:\n", df.isnull().sum())

# ------------------------------
# 3. Handle missing values
# ------------------------------
# Numerical columns: fill with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns: fill with mode (most frequent)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after handling:\n", df.isnull().sum())

# ------------------------------
# 4. Convert categorical to numerical
# ------------------------------
# Using Label Encoding for simplicity
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:\n", df.head())

# ------------------------------
# 5. Normalize / Standardize numerical features
# ------------------------------
num_cols = df.select_dtypes(include=np.number).columns

# Option 1: Standardization (Z-score normalization)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Option 2: Min-Max Normalization
# scaler = MinMaxScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nData after scaling:\n", df.head())

# ------------------------------
# 6. Visualize Outliers using Boxplots
# ------------------------------
plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, (len(num_cols) + 1) // 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# ------------------------------
# 7. Remove outliers using IQR
# ------------------------------
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in num_cols:
    df = remove_outliers_iqr(df, col)

print("\nData shape after outlier removal:", df.shape)

# ------------------------------
# Final Clean Dataset
# ------------------------------
print("\nFinal cleaned dataset:\n", df.head())
