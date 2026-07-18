"""
Train and compare 3 regression algorithms to predict restaurant
Aggregate Rating:
    1. Linear Regression
    2. Decision Tree Regression
    3. Random Forest Regression

Each is trained on the same training split and evaluated on the same
held-out test split using MSE, RMSE, MAE and R-squared. The
best-performing model (by R^2) is saved and used by the web app.

Run this once (or whenever the dataset changes) to (re)generate:
    models/rating_model.pkl        <- best model, used by app.py
    models/encoders.pkl
    models/feature_importance.png
    models/metrics.json            <- metrics for the best model (shown on the web page)
    models/model_comparison.json   <- metrics for ALL 3 models side by side
    models/model_comparison.png    <- bar chart comparing R^2 across models
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_excel(DATA_PATH)
    return df


def clean_data(df):
    df = df.copy()

    # Drop rows with no target
    df = df.dropna(subset=["Aggregate rating"])

    # Remove "Not rated" restaurants (rating = 0, no votes yet) - they add noise
    if "Rating text" in df.columns:
        df = df[df["Rating text"].str.lower() != "not rated"]

    # Fill missing cuisines
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    return df


def engineer_features(df):
    df = df.copy()
    df["Cuisine_Count"] = df["Cuisines"].apply(lambda x: len(str(x).split(",")))

    binary_map = {"Yes": 1, "No": 0}
    for col in ["Has Table booking", "Has Online delivery"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    return df


def target_encode(train_series, target, min_samples=3):
    """
    Returns a dict mapping category -> mean target value.
    Categories with very few samples fall back to the global mean
    (reduces overfitting / gives sane values for rare categories).
    """
    global_mean = target.mean()
    stats = pd.DataFrame({"cat": train_series, "target": target})
    grouped = stats.groupby("cat")["target"].agg(["mean", "count"])
    grouped["smoothed"] = np.where(
        grouped["count"] >= min_samples, grouped["mean"], global_mean
    )
    mapping = grouped["smoothed"].to_dict()
    mapping["__GLOBAL_MEAN__"] = global_mean
    return mapping


def apply_target_encoding(series, mapping):
    global_mean = mapping.get("__GLOBAL_MEAN__")
    return series.map(mapping).fillna(global_mean)


def main():
    print("Loading data...")
    df = load_data()
    print(f"Raw shape: {df.shape}")

    df = clean_data(df)
    print(f"After cleaning: {df.shape}")

    df = engineer_features(df)

    y = df["Aggregate rating"]

    # --- Target-encode high-cardinality categorical columns ---
    city_map = target_encode(df["City"], y)
    cuisines_map = target_encode(df["Cuisines"], y)

    df["City_enc"] = apply_target_encoding(df["City"], city_map)
    df["Cuisines_enc"] = apply_target_encoding(df["Cuisines"], cuisines_map)

    feature_cols = [
        "City_enc",
        "Cuisines_enc",
        "Average Cost for two",
        "Price range",
        "Has Table booking",
        "Has Online delivery",
        "Votes",
        "Cuisine_Count",
    ]

    X = df[feature_cols]
    y = df["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Define the 3 candidate algorithms ---
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=8, random_state=42
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    all_metrics = {}
    trained_models = {}

    for name, algo in candidates.items():
        print(f"\nTraining {name}...")
        algo.fit(X_train, y_train)
        preds = algo.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        all_metrics[name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        trained_models[name] = algo
        print(f"  MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    # --- Pick the best model by R^2 on the test set ---
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["r2"])
    model = trained_models[best_name]
    metrics = all_metrics[best_name]
    metrics["algorithm"] = best_name

    print(f"\nBest algorithm: {best_name}  (R2 = {metrics['r2']:.4f})")

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- Comparison bar chart: R^2 across all 3 algorithms ---
    comp_df = pd.DataFrame(all_metrics).T
    plt.figure(figsize=(8, 5))
    colors = ["#8C6E3A" if n != best_name else "#D4A24C" for n in comp_df.index]
    plt.barh(comp_df.index, comp_df["r2"], color=colors)
    plt.xlabel("R-squared (test set)")
    plt.title("Algorithm Comparison - R-squared")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"))
    plt.close()
    print("Saved model_comparison.png")

    # --- Feature importance plot (tree-based models only) ---
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
    else:
        # Linear Regression: use absolute coefficient magnitude instead
        importances = pd.Series(
            np.abs(model.coef_), index=feature_cols
        ).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#2E86AB")
    plt.title(f"Feature Importance - {best_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"))
    plt.close()
    print("Saved feature_importance.png")

    # --- Save best model + encoders needed by the web app ---
    joblib.dump(model, os.path.join(MODELS_DIR, "rating_model.pkl"))

    top_cities = df["City"].value_counts().head(50).index.tolist()
    top_cuisines = df["Cuisines"].value_counts().head(50).index.tolist()

    encoders = {
        "city_map": city_map,
        "cuisines_map": cuisines_map,
        "feature_cols": feature_cols,
        "top_cities": sorted(top_cities),
        "top_cuisines": sorted(top_cuisines),
        "price_range_options": sorted(df["Price range"].unique().tolist()),
        "cost_min": int(df["Average Cost for two"].quantile(0.01)),
        "cost_max": int(df["Average Cost for two"].quantile(0.99)),
    }
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))

    print("\n=== Summary: all 3 algorithms on the test set ===")
    for name, m in all_metrics.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"{name:28s} R2={m['r2']:.3f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}{marker}")

    print(f"\nDone. Best model ({best_name}) + encoders saved in /models")


if __name__ == "__main__":
    main()
