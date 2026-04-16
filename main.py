from flask import Flask, request
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route('/')
def home():
    return "Welcome to the MLOps Lab!"

@app.route('/predict')
def predict():
    X = float(request.args.get('x'))
    prediction = model.predict(X)
    return {"prediction": prediction[0]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)