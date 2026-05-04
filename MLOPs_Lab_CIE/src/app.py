from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model (use same logic as CLI — train again if needed)
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Load data and retrain model (since we removed model.pkl)
data = pd.read_csv("../data/training_data.csv")

X = data.drop("completion_days", axis=1)
y = data["completion_days"]

model = GradientBoostingRegressor()
model.fit(X, y)

# Input schema
class InputData(BaseModel):
    course_hours: float
    quizzes_count: float
    difficulty_level: float
    learner_experience: float


@app.get("/health")
def health():
    return {"alive": True, "service": "EduTrack completion_days API"}


@app.post("/predict")
def predict(data: InputData):
    features = np.array([[ 
        data.course_hours,
        data.quizzes_count,
        data.difficulty_level,
        data.learner_experience
    ]])

    prediction = model.predict(features)[0]

    return {"prediction": float(prediction)}