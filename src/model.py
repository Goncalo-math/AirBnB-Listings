from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

BEST_FEATURES = [
    "room_type",               # most important by far
    "property_type",           # second most important
    "city",                    # strong importance
    "accommodates",            # high correlation with price (drop bedrooms)
    "review_scores_rating",    # mid importance
    "minimum_nights",          # mid importance
    "latitude",                # mid importance
    # DROPPED: neighbourhood, longitude, district, host_is_superhost,
    #          instant_bookable, review_scores_location, bedrooms
]
TARGET = "price_log"

def train(dfs):

    df = dfs["airbnb_listings"]  # unpack the dict

    df.columns = df.columns.str.strip()  # remove leading/trailing whitespace from column names

    # Remove non-numeric prices
    mask = pd.to_numeric(df["price"], errors="coerce").isna()  # True for non-numeric values
    print(f"Removing {mask.sum()} non-numeric price rows")     # log how many rows are removed
    df = df[~mask].copy()                                      # keep only rows with numeric prices

    
    print(f'Zero price rows: {(df["price"] == 0).sum()}')      # log how many zero price rows are present
    df = df[df['price'] > 0]                                   # remove zero price rows                  

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
    lr_results = evaluate_model(
    'Linear Regression',
    LinearRegression(),
    X_train, X_test, y_train, y_test
    )
    print(f"CV R² Mean: {lr_results['CV_R2']:.4f} ± {lr_results['R2']:.4f}")

    # Train with Random Forest (added complexity, may improve performance but slower)
    rf_results = evaluate_model(
    'Random Forest',
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
    )

    # Train for LightGBM (added complexity, may improve performance but slower)
    lgbm_results = evaluate_model(
    'LightGBM',
    LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    X_train, X_test, y_train, y_test
    )


    # Get feature importance
    lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)  # make sure it's fitted

    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgbm_model.feature_importances_
    }).sort_values('importance', ascending=True)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(importance['feature'], importance['importance'], color='steelblue')
    plt.title('LightGBM Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


    numeric_features = [
    "latitude", "longitude", "accommodates", "bedrooms",
    "minimum_nights", "review_scores_rating", "review_scores_location",
    "host_is_superhost", "instant_bookable", "price_log"
]

    corr = df[numeric_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return lr_results, rf_results, lgbm_results

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

    print(f'\n--- {name} ---')
    print(f'MAE:            {mae:.4f}')
    print(f'RMSE:           {rmse:.4f}')
    print(f'R²:             {r2:.4f}')
    print(f'CV R² (10-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_scores.mean()}
