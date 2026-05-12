from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np

FEATURES = [
    "latitude", "longitude",                  # location
    "neighbourhood", "district", "city",      # location (categorical)
    "room_type", "property_type",             # type
    "accommodates", "bedrooms",               # size
    "minimum_nights",                         # restrictions
    "host_is_superhost",                      # host quality
    "review_scores_rating", "review_scores_location",
    "instant_bookable"
]
TARGET = "price_log"

def train(dfs):

    df = dfs["airbnb_listings"]  # unpack the dict

    df.columns = df.columns.str.strip()  # remove leading/trailing whitespace from column names

    # Remove non-numeric prices
    mask = pd.to_numeric(df["price"], errors="coerce").isna()  # True for non-numeric values
    print(f"Removing {mask.sum()} non-numeric price rows")     # log how many rows are removed
    df = df[~mask].copy()                                      # keep only rows with numeric prices

    # df["price"] = df["price"].astype(float)
    # df = df[df["price"] > 0]
    df[TARGET] = np.log1p(df["price"])

    # Encode bool columns
    bool_cols = ["host_is_superhost", "instant_bookable"]
    df[bool_cols] = df[bool_cols].apply(lambda col: col.map({"t": 1, "f": 0}))

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    # Apply OrdinalEncoder via ColumnTransformer
    cat_cols = ["neighbourhood", "district", "city", "room_type", "property_type"]
    ct = ColumnTransformer(transformers=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough')

    X_encoded = ct.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42  # no stratify for regression
    )

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    # Regression metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\nTest Set Performance for Linear Regression:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Cross-validation
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=10, scoring="r2")
    print(f"CV R² Mean: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

    # Train with Random Forest (added complexity, may improve performance but slower)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    # Regression metrics for Random Forest
    mae_rf  = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf   = r2_score(y_test, y_pred_rf)

    print("\nTest Set Performance for Random Forest:")
    print(f"MAE:  {mae_rf:.4f}")
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"R²:   {r2_rf:.4f}")

    # Cross-validation
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring="r2")
    print(f"CV R² Mean: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

    # Train for LightGBM (added complexity, may improve performance but slower)
    lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)

    y_pred_lgbm = lgbm_model.predict(X_test)

    # Regression metrics for LightGBM
    mae_lgbm  = mean_absolute_error(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    r2_lgbm   = r2_score(y_test, y_pred_lgbm)

    print("\nTest Set Performance for LightGBM:")
    print(f"MAE:  {mae_lgbm:.4f}")
    print(f"RMSE: {rmse_lgbm:.4f}")
    print(f"R²:   {r2_lgbm:.4f}")

    # Instead of X_train (numpy array), keep it as DataFrame
    X_train_df = pd.DataFrame(X_train, columns=X.columns)

    lgbm_scores = cross_val_score(lgbm_model, X_train_df, y_train, cv=10, scoring="r2")
    print(f"CV R² Mean: {lgbm_scores.mean():.4f} ± {lgbm_scores.std():.4f}")

    return lr_model