import joblib
model = joblib.load("model/model.pkl")

X_test = [[5]]
prediction = model.predict(X_test)

assert prediction[0] == 10, "Test Failed"

print("Test passed: Prediction is correct.")