import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load data
data = pd.read_csv("data/training_data.csv")

X = data[["course_hours", "quizzes_count", "difficulty_level", "learner_experience"]]
y = data["completion_days"]

# Train model
model = GradientBoostingRegressor()
model.fit(X, y)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--course_hours", type=float, required=True)
parser.add_argument("--quizzes_count", type=float, required=True)
parser.add_argument("--difficulty_level", type=float, required=True)
parser.add_argument("--learner_experience", type=float, required=True)

args = parser.parse_args()

# Input
input_data = [[
    args.course_hours,
    args.quizzes_count,
    args.difficulty_level,
    args.learner_experience
]]

# Predict
prediction = model.predict(input_data)[0]

print("Prediction:", prediction)