import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
import math

from preprocess import load_and_clean, get_feature_matrix

DATA_PATH   = os.path.join("data", "Nassau_Candy_Distributor.csv")
MODELS_DIR  = "models"

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  {name:<30} RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2}


def train():
    print("=" * 55)
    print("   Nassau Candy — ML Training")
    print("=" * 55)

    # ── Load & prepare data ──────────────────────────────────
    df, encoders = load_and_clean(DATA_PATH)
    X, y, feature_cols = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Train rows: {len(X_train)}  |  Test rows: {len(X_test)}\n")

    # ── Train 3 models ───────────────────────────────────────
    candidates = [
        ("Linear Regression",    LinearRegression()),
        ("Random Forest",        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting",    GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]

    print("📈 Model Results:")
    results = []
    for name, model in candidates:
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    # ── Pick best model (lowest RMSE) ────────────────────────
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n🏆 Best model: {best['name']}  (RMSE={best['rmse']:.1f})")

    # ── Save everything ──────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best["model"],  os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(feature_cols,   os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(encoders,       os.path.join(MODELS_DIR, "encoders.pkl"))

    # Save results summary
    summary = pd.DataFrame([
        {"Model": r["name"], "RMSE": round(r["rmse"],2),
         "MAE": round(r["mae"],2), "R2": round(r["r2"],3)}
        for r in results
    ])
    summary.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)

    print("\n💾 Saved:")
    print("   models/best_model.pkl")
    print("   models/feature_cols.pkl")
    print("   models/encoders.pkl")
    print("   models/model_results.csv")
    print("\n✅ Training complete!")
    return best["model"], encoders, feature_cols


if __name__ == "__main__":
    train()