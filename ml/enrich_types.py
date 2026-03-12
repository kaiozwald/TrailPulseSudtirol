import pandas as pd
import numpy as np
import pickle
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("activities_clean.csv")

# ── Step 1: Define known good records as training data ──
# These are records where we trust the activity type label
KNOWN_TYPES = ["hiking", "cycling", "climbing", "skiing", "running", "family", "winter", "summer"]

train_df = df[df["activity_type"].isin(KNOWN_TYPES)].copy()
print(f"Known typed records for training: {len(train_df)}")
print(train_df["activity_type"].value_counts())

# ── Step 2: Train a type classifier ──
FEATURES = [
    "distance_m", "duration_min", "altitude_diff", "altitude_start",
    "has_rentals", "lift_available", "is_open", "is_prepared"
]

X_train = train_df[FEATURES]
y_train = train_df["activity_type"]

# Use FLAML again for consistency
type_clf = AutoML()
type_clf.fit(
    X_train, y_train,
    task="classification",
    time_budget=60,
    metric="macro_f1",
    verbose=1
)

print(f"\nBest type classifier: {type_clf.best_estimator}")

# Quick eval on training data
y_pred = type_clf.predict(X_train)
print("\nType Classifier Report (train set):")
print(classification_report(y_train, y_pred))

# ── Step 3: Predict types for ALL records ──
X_all = df[FEATURES]
df["activity_type_predicted"] = type_clf.predict(X_all)
df["type_prediction_confidence"] = type_clf.predict_proba(X_all).max(axis=1)

# ── Step 4: Enrich — use predicted type where original is general/other/unknown ──
mask = df["activity_type"].isin(["general", "other"])
df.loc[mask, "activity_type"] = df.loc[mask, "activity_type_predicted"]

print(f"\nActivity type distribution after enrichment:")
print(df["activity_type"].value_counts())

# ── Step 5: Re-encode activity type with new labels ──
from sklearn.preprocessing import LabelEncoder
le_type_new = LabelEncoder()
df["activity_type_enc"] = le_type_new.fit_transform(df["activity_type"])

# Save updated encoder
with open("encoder_type.pkl", "wb") as f:
    pickle.dump(le_type_new, f)

# Save type classifier
with open("type_classifier.pkl", "wb") as f:
    pickle.dump(type_clf, f)

# Save enriched dataset
df.to_csv("activities_enriched.csv", index=False)
print(f"\nEnriched dataset saved — {len(df)} records")
print(f"Type prediction confidence avg: {df['type_prediction_confidence'].mean():.3f}")