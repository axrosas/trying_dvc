# scripts/train_model.py
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/processed_data.csv")

y = df["Survived"]
X = df.drop("Survived", axis=1)

model = LinearRegression()
model.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(model, "models/model1.pkl")
