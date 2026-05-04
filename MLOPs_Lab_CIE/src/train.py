import pandas as pd
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("data/training_data.csv")

X = data.drop("completion_days", axis=1)
y = data["completion_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("edutrack-completion-days")

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = []

best_rmse = float("inf")
best_model_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = (abs((y_test - y_pred) / y_test).mean()) * 100

        # Log to MLflow
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        mlflow.set_tag("experiment_type", "baseline_comparison")

        mlflow.sklearn.log_model(model, name)

        # Save results
        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        # Best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

# Save best model
import joblib
joblib.dump(best_model, "model.pkl")

# Save JSON output
output = {
    "experiment_name": "edutrack-completion-days",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 complete ✅")