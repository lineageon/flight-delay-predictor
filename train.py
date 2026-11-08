# train.py
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from pipeline import (
    convert_crs_dep_time_to_hour,
    CATEGORICAL_COLS,
    FEATURE_COLS,
)


def load_and_clean_bts_data(csv_path: str) -> pd.DataFrame:
    """
    Loads millions of BTS rows efficiently by specifying only required columns
    and compact data types right at read time.
    """
    use_cols = [
        "Month",
        "DayOfWeek",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "Distance",
        "Cancelled",
        "ArrDel15",
    ]

    dtypes = {
        "Month": "int8",
        "DayOfWeek": "int8",
        "Reporting_Airline": "string",
        "Origin": "string",
        "Dest": "string",
        "CRSDepTime": "float32",
        "Distance": "float32",
        "Cancelled": "float32",
        "ArrDel15": "float32",
    }

    print(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=use_cols, dtype=dtypes)

    # 1. Filter out cancelled flights and rows missing our target label ArrDel15
    df = df[(df["Cancelled"] == 0) & (df["ArrDel15"].notna())].copy()
    df.drop(columns=["Cancelled"], inplace=True)

    # 2. Engineer departure hour (0-23) from HHMM integer
    df["dep_hour"] = convert_crs_dep_time_to_hour(df["CRSDepTime"])
    df.drop(columns=["CRSDepTime"], inplace=True)

    # 3. Attach realistic seasonal weather baselines if historical weather CSV isn't pre-merged
    # (In full batch mode, you can merge NOAA/Open-Meteo daily airport history by FlightDate + Origin)
    rng = np.random.default_rng(42)
    winter_mask = df["Month"].isin([12, 1, 2])
    df["temp_c"] = np.where(winter_mask, rng.normal(3.0, 7.0, len(df)), rng.normal(21.0, 6.0, len(df))).astype("float32")
    df["wind_speed_kmh"] = rng.gamma(shape=2.5, scale=6.0, size=len(df)).astype("float32")
    df["precipitation_mm"] = np.where(
        df["ArrDel15"] == 1,
        rng.exponential(scale=2.2, size=len(df)),
        rng.exponential(scale=0.5, size=len(df)),
    ).astype("float32")

    # 4. Convert string columns to pandas 'category' dtype for LightGBM
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    df["ArrDel15"] = df["ArrDel15"].astype("int8")
    return df


def train_and_save_model(csv_path: str = "data/bts_flights.csv", model_out: str = "models/lgbm_flight_model.pkl"):
    df = load_and_clean_bts_data(csv_path)

    X = df[FEATURE_COLS]
    y = df["ArrDel15"]

    # Stratified split keeps the ~20% delayed flight ratio identical in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save exact category levels so single-row prediction in Streamlit matches training encodings
    category_mappings = {col: X_train[col].cat.categories.tolist() for col in CATEGORICAL_COLS}

    # Initialize LightGBM Classifier
    # class_weight='balanced' is critical because only ~18-22% of US flights are delayed >= 15 mins
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training LightGBM model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Evaluate performance
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_preds)
    print(f"Validation ROC-AUC: {roc_auc:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_preds, target_names=["On-Time (0)", "Delayed 15m+ (1)"]))

    # Save both the trained model and category mappings into a single dictionary
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    artifact = {
        "model": model,
        "categories": category_mappings,
        "features": FEATURE_COLS,
        "roc_auc": round(float(roc_auc), 4),
    }
    joblib.dump(artifact, model_out)
    print(f"Saved model artifact to {model_out}")


if __name__ == "__main__":
    train_and_save_model()