"""
CODTECH Data Science Internship
Task 1: Data Pipeline Development (ETL)

Objective:
Create an automated ETL pipeline for data preprocessing,
transformation, and loading using Pandas and Scikit-learn.

Author: Shamim Akthar
"""

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# 1. EXTRACT (DATA COLLECTION)
# =========================
data = {
    'Age': [25, 30, np.nan, 35, 40, np.nan, 22, 28],
    'Salary': [50000, 60000, 55000, np.nan, 65000, 58000, 45000, 52000],
    'City': ['New York', 'Paris', 'London', 'New York', 
             'Paris', 'London', 'New York', 'Paris'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\n----------------------------------\n")


# =========================
# 2. DEFINE FEATURES & TARGET
# =========================
X = df.drop('Purchased', axis=1)
y = df['Purchased']


# =========================
# 3. TRANSFORM (PREPROCESSING PIPELINE)
# =========================

# Numeric feature processing
numeric_features = ['Age', 'Salary']
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical feature processing
categorical_features = ['City']
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])


# =========================
# 4. LOAD (APPLY ETL PIPELINE)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("ETL Pipeline Applied Successfully!")
print("Processed Training Data Shape:", X_train_processed.shape)
print("Processed Testing Data Shape:", X_test_processed.shape)

print("\nTask 1 Completed Successfully ")
