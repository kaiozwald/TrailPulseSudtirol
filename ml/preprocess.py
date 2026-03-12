import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("activities.csv")

# ── 1. Convert duration from hours to minutes ──
df["duration_min"] = df["duration_min"] * 60

# ── 2. Clean activity types — map GUIDs to readable labels ──
type_map = {
    "6285F49DBBE04393BAD29E6EF219EB03": "hiking",
    "978F89296ACB4DB4B6BD1C269341802F": "cycling",
    "2A289E6E6C56417C8C2A895CED9F07DB": "skiing",
    "9CBAC00246A8467E93DD66F3A1A9C594": "running",
    "2B0E6F774C284C4B8B51B666AC36E579": "climbing",
    "activity": "general",
    "other hikes": "hiking",
    "family hikings": "family",
    "winter": "winter",
    "summer": "summer",
}
df["activity_type"] = df["activity_type"].map(type_map).fillna("other")

# ── 3. Impute missing numerical values with median ──
for col in ["distance_m", "duration_min", "altitude_diff", "altitude_start"]:
    median = df[col].median()
    df[col] = df[col].fillna(median)
    print(f"Imputed {col} with median: {median:.2f}")

# ── 4. Handle difficulty — fill missing with mode, encode numerically ──
df["difficulty"] = df["difficulty"].replace("", np.nan)
df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
df["difficulty"] = df["difficulty"].fillna(df["difficulty"].mode()[0])
df["difficulty"] = df["difficulty"].astype(int)

# ── 5. Encode categorical columns ──
le_type = LabelEncoder()
le_location = LabelEncoder()
df["activity_type_enc"] = le_type.fit_transform(df["activity_type"])
df["location_enc"] = le_location.fit_transform(df["location"])

# ── 6. Save encoders for later use in the recommender ──
with open("ml/encoder_type.pkl", "wb") as f:
    pickle.dump(le_type, f)
with open("ml/encoder_location.pkl", "wb") as f:
    pickle.dump(le_location, f)

# ── 7. Save cleaned dataset ──
df.to_csv("ml/activities_clean.csv", index=False)

print(f"\nCleaned dataset shape: {df.shape}")
print(f"\nActivity type distribution after cleaning:")
print(df["activity_type"].value_counts())
print(f"\nDifficulty distribution after cleaning:")
print(df["difficulty"].value_counts())
print(f"\nMissing values remaining:")
print(df.isnull().sum())