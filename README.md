# House Price Prediction

A Linear Regression model that predicts house sale prices from key property features. Built as Project 1 for the Syntecxhub internship.

## Overview

This project loads housing data, cleans it, trains a Linear Regression model, and evaluates how well it predicts sale prices.

**Steps performed:**
1. Load and explore the dataset
2. Clean data (handle missing values, remove outliers)
3. Select features, split into train/test sets
4. Train a Linear Regression model
5. Evaluate using RMSE and R², and interpret coefficients
6. Save the trained model and generate an example prediction

## Dataset

[Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (`train.csv`)

> Note: `train.csv` is not included in this repo due to size/licensing — download it from the link above and place it in the project folder before running the script.

**Features used** (mapped from the original Kaggle columns):

| Feature          | Source column              | Description                          |
|------------------|-----------------------------|---------------------------------------|
| `sqft`           | `GrLivArea`                 | Above-ground living area (sq ft)      |
| `bedrooms`       | `BedroomAbvGr`               | Bedrooms above ground                 |
| `bathrooms`      | `FullBath`                   | Full bathrooms                        |
| `age`            | `YrSold - YearBuilt`         | House age at time of sale             |
| `location_score` | `OverallQual`                 | Overall quality rating (1–10)         |
| `price` (target) | `SalePrice`                   | Final sale price                      |

## How to Run

```bash
pip install numpy pandas scikit-learn joblib
python house_price_prediction.py
```

Make sure `train.csv` is in the same folder as the script before running.

## Results

| Metric | Value      |
|--------|------------|
| RMSE   | ~$26,913   |
| R²     | ~0.77      |

The model explains about **77% of the variance** in house sale prices — a solid result for a simple 5-feature linear model on real-world data.

### Coefficient Interpretation

- **Square footage** and **overall quality** are the strongest positive drivers of price.
- **Age** has the strongest negative effect — older homes sell for less, all else equal.
- **Bedrooms** shows a slight *negative* coefficient. This isn't a bug: since square footage is already in the model, an extra bedroom at a fixed size usually means smaller individual rooms, which buyers value less. The model is capturing "room count relative to size," not "bigger house."

## Output

Running the script prints data-cleaning summary stats, evaluation metrics, and coefficient interpretation to the console, and saves:
- `house_price_model.pkl` — the trained model
- `scaler.pkl` — the fitted feature scaler (needed to preprocess new data the same way before predicting)

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn (LinearRegression, StandardScaler, train_test_split)
- joblib (model persistence)

## Author

Krishna Shukla — Syntecxhub Internship
