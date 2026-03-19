from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    day = int(request.form['day'])
    festival = int(request.form['festival'])
    weather = int(request.form['weather'])
    customers = int(request.form['customers'])
    prev_day = int(request.form['prev_day'])
    prev_week = int(request.form['prev_week'])

    weekend = 1 if day >= 5 else 0

    features = pd.DataFrame([[day, festival, weather, customers, prev_day, prev_week, weekend]],
    columns=[
        'Day_of_Week',
        'Festival',
        'Weather',
        'Expected_Customers',
        'Previous_Day_Consumption',
        'Previous_Week_Same_Day',
        'Weekend'
    ])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text="Recommended Meals to Prepare: {}".format(int(prediction[0]))
    )

if __name__ == "__main__":
    app.run(debug=True)