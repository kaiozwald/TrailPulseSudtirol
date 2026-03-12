import pandas as pd
import pickle
import numpy as np

# ── Load model and encoders ──
with open("ml/model.pkl", "rb") as f:
    automl = pickle.load(f)
with open("ml/encoder_type.pkl", "rb") as f:
    le_type = pickle.load(f)
with open("ml/encoder_location.pkl", "rb") as f:
    le_location = pickle.load(f)

df = pd.read_csv("ml/activities_enriched.csv")

FEATURES = [
    "distance_m", "duration_min", "altitude_diff", "altitude_start",
    "has_rentals", "lift_available", "is_open", "is_prepared",
    "activity_type_enc", "location_enc"
]

def recommend(
    preferred_type="hiking",
    max_distance_m=5000,
    max_duration_min=120,
    wants_rentals=False,
    only_open=True,
    top_n=5
):
    """
    Returns top N activity recommendations matching the user profile.
    
    Args:
        preferred_type: activity type (hiking, cycling, skiing, running, climbing, family)
        max_distance_m: maximum route distance in metres
        max_duration_min: maximum duration in minutes
        wants_rentals: whether rental equipment is needed
        only_open: only return currently open activities
        top_n: number of recommendations to return
    """

    filtered = df.copy()

    # ── Only use records with real (non-imputed) data ──
    filtered = filtered[
        (filtered["distance_m"] != 900.0) |
        (filtered["duration_min"] != 4.2) |
        (filtered["altitude_diff"] != 183.0)
    ]

    # ── Hard filters ──
    if only_open:
        filtered = filtered[filtered["is_open"] == 1]
    if wants_rentals:
        filtered = filtered[filtered["has_rentals"] == 1]
    filtered = filtered[filtered["distance_m"] <= max_distance_m]
    filtered = filtered[filtered["duration_min"] <= max_duration_min]

    # ── Activity type filter ──
    known_types = list(le_type.classes_)
    if preferred_type in known_types:
        filtered = filtered[filtered["activity_type"] == preferred_type]

    if filtered.empty:
        print("No activities match the given filters.")
        return pd.DataFrame()

    # ── Predict and score ──
    X = filtered[FEATURES]
    filtered = filtered.copy()
    filtered["predicted_difficulty"] = automl.predict(X)
    proba = automl.predict_proba(X)
    filtered["confidence"] = proba.max(axis=1)

    # ── Weighted score: confidence + open bonus + rentals bonus ──
    filtered["score"] = (
        filtered["confidence"] * 0.6 +
        filtered["is_open"] * 0.2 +
        filtered["is_prepared"] * 0.1 +
        filtered["has_rentals"] * 0.1
    )

    # ── Only use records with real non-zero route data ──
    filtered = filtered[
        (filtered["distance_m"] > 0) &
        (filtered["duration_min"] > 0) &
        (filtered["altitude_diff"] > 0)
    ]

    results = filtered.sort_values("score", ascending=False).head(top_n)
    return results[["title", "activity_type", "location", "distance_m",
                     "duration_min", "altitude_diff", "predicted_difficulty", "score"]]

if __name__ == "__main__":
    for activity in ["hiking", "cycling", "skiing", "climbing", "running", "family"]:
        print(f"\n=== {activity.upper()} ===")
        r = recommend(preferred_type=activity, max_distance_m=10000,
                      max_duration_min=180, only_open=False)
        if not r.empty:
            print(r.head(3).to_string(index=False))
        else:
            print("No results")