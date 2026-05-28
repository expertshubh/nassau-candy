import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── 1. FACTORY MAPPING ──────────────────────────────────────────────────────
# Which product comes from which factory
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":    "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":            "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":       "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":           "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":    "Wicked Choccy's",
    "Laffy Taffy":                          "Sugar Shack",
    "SweeTARTS":                            "Sugar Shack",
    "Nerds":                                "Sugar Shack",
    "Fun Dip":                              "Sugar Shack",
    "Fizzy Lifting Drinks":                 "Sugar Shack",
    "Everlasting Gobstopper":               "Secret Factory",
    "Lickable Wallpaper":                   "Secret Factory",
    "Wonka Gum":                            "Secret Factory",
    "Hair Toffee":                          "The Other Factory",
    "Kazookles":                            "The Other Factory",
}

# Factory GPS coordinates (latitude, longitude)
FACTORY_COORDS = {
    "Lot's O' Nuts":      (32.881893, -111.768036),
    "Wicked Choccy's":    (32.076176,  -81.088371),
    "Sugar Shack":        (48.119140,  -96.181150),
    "Secret Factory":     (41.446333,  -90.565487),
    "The Other Factory":  (35.117500,  -89.971107),
}

# Customer region approximate center coordinates
REGION_COORDS = {
    "Atlantic":  (38.9,  -77.0),
    "Gulf":      (29.7,  -90.1),
    "Interior":  (41.8,  -87.6),
    "Pacific":   (34.0, -118.2),
}

def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in miles between two GPS points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_and_clean(filepath):
    """Load CSV, clean it, and add useful columns."""
    print("📂 Loading data...")
    df = pd.read_csv(filepath)

    # Parse dates (format: DD-MM-YYYY)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)

    # ── 2. DERIVE LEAD TIME (target variable for ML) ────────────────────────
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    print(f"   Lead Time range: {df['Lead Time'].min()} – {df['Lead Time'].max()} days")

    # Remove impossible lead times (negative or extreme outliers)
    df = df[df['Lead Time'] >= 0]
    df = df[df['Lead Time'] <= df['Lead Time'].quantile(0.99)]

    # ── 3. MAP FACTORIES ─────────────────────────────────────────────────────
    df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY)

    # ── 4. SHIPPING DISTANCE FEATURE ─────────────────────────────────────────
    def get_distance(row):
        f_lat, f_lon = FACTORY_COORDS[row['Factory']]
        r_lat, r_lon = REGION_COORDS.get(row['Region'], (39.5, -98.3))
        return haversine_miles(f_lat, f_lon, r_lat, r_lon)

    df['Shipping Distance (miles)'] = df.apply(get_distance, axis=1)

    # ── 5. ENCODE CATEGORICAL COLUMNS ────────────────────────────────────────
    encoders = {}
    for col in ['Product Name', 'Factory', 'Region', 'Ship Mode']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   Encoded '{col}': {list(le.classes_)}")

    print(f"\n✅ Data ready! {len(df)} rows, {df.shape[1]} columns.")
    return df, encoders


def get_feature_matrix(df):
    """Return the feature columns (X) and target column (y) for ML."""
    feature_cols = [
        'Product Name_enc',
        'Factory_enc',
        'Region_enc',
        'Ship Mode_enc',
        'Shipping Distance (miles)',
        'Units',
        'Cost',
    ]
    X = df[feature_cols]
    y = df['Lead Time']
    return X, y, feature_cols


if __name__ == "__main__":
    # Test it by running: python src/preprocess.py
    filepath = os.path.join("data", "Nassau_Candy_Distributor.csv")
    df, encoders = load_and_clean(filepath)

    X, y, feature_cols = get_feature_matrix(df)
    print("\n📊 Feature matrix shape:", X.shape)
    print("🎯 Target (Lead Time) average:", round(y.mean(), 1), "days")
    print("\nFirst 3 rows of features:")
    print(X.head(3).to_string())

    # Save encoders for later use in the app
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, os.path.join("models", "encoders.pkl"))
    print("\n💾 Encoders saved to models/encoders.pkl")
