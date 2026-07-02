"""
Project 1: House Price Prediction
-----------------------------------
A complete Linear Regression pipeline:
1. Load & explore data
2. Clean data
3. Select features, split train/test
4. Train a Linear Regression model
5. Evaluate with RMSE and R², interpret coefficients
6. Save the model and show example predictions

NOTE: This script generates a realistic synthetic housing dataset so it
runs immediately with no downloads. For your actual submission, replace
the `load_data()` function with a real dataset, e.g.:
    - Kaggle "House Prices - Advanced Regression Techniques"
    - sklearn.datasets.fetch_california_housing()
    - Any housing CSV with columns like: sqft, bedrooms, bathrooms,
      age, location_score, price
Just point pd.read_csv() at your file and keep the rest of the
pipeline the same.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data():
    """Load the real Kaggle 'House Prices - Advanced Regression Techniques'
    dataset (train.csv) and map its columns to our simple feature set.
    """
    raw = pd.read_csv("train.csv")

    df = pd.DataFrame({
        "sqft": raw["GrLivArea"],
        "bedrooms": raw["BedroomAbvGr"],
        "bathrooms": raw["FullBath"],
        "age": raw["YrSold"] - raw["YearBuilt"],          # house age at sale
        "location_score": raw["OverallQual"],              # 1-10 quality rating, stands in for location
        "price": raw["SalePrice"],
    })
    return df


# ----------------------------------------------------------------------
# 2. CLEAN DATA
# ----------------------------------------------------------------------
def clean_data(df):
    print("Missing values before cleaning:\n", df.isnull().sum())

    # Fill missing numeric values with the median
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Remove extreme outliers using the IQR method on price
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df["price"] >= lower) & (df["price"] <= upper)].reset_index(drop=True)
    print(f"Removed {before - len(df)} outlier row(s).")

    print("\nBasic stats:\n", df.describe())
    return df


# ----------------------------------------------------------------------
# 3. SELECT FEATURES & SPLIT
# ----------------------------------------------------------------------
def prepare_features(df):
    features = ["sqft", "bedrooms", "bathrooms", "age", "location_score"]
    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (helps interpret coefficients on a common scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, features


# ----------------------------------------------------------------------
# 4. TRAIN MODEL
# ----------------------------------------------------------------------
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# 5. EVALUATE
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, features):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nRMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    print("\nCoefficients (scaled features, so magnitudes are comparable):")
    for feat, coef in zip(features, model.coef_):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feat:16s}: {coef:>12,.2f}  -> price {direction} as this rises")

    print(f"\nIntercept: {model.intercept_:,.2f}")
    return y_pred, rmse, r2


# ----------------------------------------------------------------------
# 6. SAVE MODEL & EXAMPLE PREDICTIONS
# ----------------------------------------------------------------------
def save_and_predict_example(model, scaler, features):
    joblib.dump(model, "house_price_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved model to house_price_model.pkl")

    # Example: predict price for a new house
    example = pd.DataFrame([{
        "sqft": 2200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 10,
        "location_score": 7.5,
    }])[features]

    example_scaled = scaler.transform(example)
    predicted_price = model.predict(example_scaled)[0]
    print(f"\nExample prediction for a new house:\n{example.to_string(index=False)}")
    print(f"Predicted price: ${predicted_price:,.2f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler, features = prepare_features(df)
    model = train_model(X_train_s, y_train)
    evaluate_model(model, X_test_s, y_test, features)
    save_and_predict_example(model, scaler, features)
