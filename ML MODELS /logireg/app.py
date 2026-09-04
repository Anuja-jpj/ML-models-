from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    credit_score = int(request.form["credit_score"])
    geography = request.form["geography"]
    gender = request.form["gender"]
    age = int(request.form["age"])
    tenure = int(request.form["tenure"])
    balance = float(request.form["balance"])
    num_products = int(request.form["num_products"])
    has_card = int(request.form["has_card"])
    active_member = int(request.form["active_member"])
    estimated_salary = float(request.form["estimated_salary"])

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [active_member],
        "EstimatedSalary": [estimated_salary]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    probability = round(probability * 100, 2)

    if prediction == 1:
        result = "Customer is likely to EXIT ❌"
    else:
        result = "Customer is likely to STAY ✅"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)