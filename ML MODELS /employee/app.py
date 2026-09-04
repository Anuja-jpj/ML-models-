from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------------
# Load the trained ML model
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "employee_income_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("employee_income_model loaded successfully!")


# ---------------------------------------
# Home Page
# ---------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------
# Prediction
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        age = float(request.form["age"])
        experience = float(request.form["experience"])

        # Create input for model
        input_data = np.array([[age, experience]])

        # Make prediction
        prediction = model.predict(input_data)

        # Convert prediction to number
        predicted_income = round(float(prediction[0]), 2)

        # Send result back to HTML
        return render_template(
            "index.html",
            prediction=predicted_income,
            age=age,
            experience=experience
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ---------------------------------------
# Run Flask Application
# ---------------------------------------

if __name__ == "__main__":
    app.run(debug=True)