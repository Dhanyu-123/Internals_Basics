import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

# Load dataset
data = pd.read_csv("data/training_data.csv")

# Features and target
X = data.drop("completion_days", axis=1)
y = data["completion_days"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    }

# Save results
os.makedirs("../results", exist_ok=True)

with open("../results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Training complete. Results saved.")