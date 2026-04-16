import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([[1], [2], [3], [4]])
Y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, Y)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully.")