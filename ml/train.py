import pandas as pd
import pickle
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("activities_enriched.csv")

# ── Features and target ──
FEATURES = [
    "distance_m", "duration_min", "altitude_diff", "altitude_start",
    "has_rentals", "lift_available", "is_open", "is_prepared",
    "activity_type_enc", "location_enc"
]
TARGET = "difficulty"

X = df[FEATURES]
y = df[TARGET]

# ── Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ── FLAML AutoML ──
automl = AutoML()
automl.fit(
    X_train, y_train,
    task="classification",
    time_budget=60,        # 60 seconds — increase if you want better results
    metric="macro_f1",     # handles class imbalance better than accuracy
    verbose=1
)

print(f"\nBest model: {automl.best_estimator}")
print(f"Best config: {automl.best_config}")

# ── Evaluate ──
y_pred = automl.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Easy(2)", "Medium(4)", "Hard(6)"]))

# ── Save model ──
with open("model.pkl", "wb") as f:
    pickle.dump(automl, f)
print("\nModel saved to model.pkl")