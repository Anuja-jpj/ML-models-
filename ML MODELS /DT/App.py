from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "DTModel.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# FEATURE ORDER
# Must match the order used while training the model
# ---------------------------------------------------------

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data from HTML
        data = request.get_json()

        print("\n===================================")
        print("PREDICT REQUEST RECEIVED")
        print("===================================")
        print("Received data:")
        print(data)

        # Check whether data was received
        if not data:
            return jsonify({
                "error": "No data received."
            }), 400

        # -------------------------------------------------
        # CHECK FOR MISSING VALUES
        # -------------------------------------------------

        missing = []

        for feature in FEATURE_ORDER:

            if feature not in data or data[feature] == "":
                missing.append(feature)

        if missing:

            return jsonify({
                "error": "Missing values: " + ", ".join(missing)
            }), 400


        # -------------------------------------------------
        # CONVERT INPUT VALUES TO NUMBERS
        # -------------------------------------------------

        features = []

        for feature in FEATURE_ORDER:

            value = float(data[feature])
            features.append(value)


        # -------------------------------------------------
        # CREATE DATAFRAME
        # This avoids the sklearn feature-name warning
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_ORDER
        )

        print("\nData sent to model:")
        print(input_data)


        # -------------------------------------------------
        # MAKE PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )


        # -------------------------------------------------
        # CREATE RESULT
        # -------------------------------------------------

        prediction = int(prediction)

        if prediction == 1:

            label = "Heart Disease Detected"

        else:

            label = "No Heart Disease Detected"


        result = {
            "prediction": prediction,
            "label": label,
            "confidence": confidence
        }


        # -------------------------------------------------
        # PRINT RESULT IN TERMINAL
        # -------------------------------------------------

        print("\n===================================")
        print("PREDICTION RESULT")
        print("===================================")
        print("Prediction:", prediction)
        print("Label:", label)
        print("Confidence:", confidence, "%")
        print("===================================\n")


        # Send result back to browser
        return jsonify(result)


    except ValueError as error:

        print("VALUE ERROR:", error)

        return jsonify({
            "error": "Please enter valid numerical values."
        }), 400


    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


# ---------------------------------------------------------
# RUN FLASK
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)