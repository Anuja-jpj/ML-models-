from flask import Flask, render_template, request
from pathlib import Path
import joblib

app = Flask(__name__)

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Model and vectorizer paths
MODEL_PATH = BASE_DIR / "naive_bayes_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

# Load trained model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully!")

except Exception as e:
    print("Error loading model/vectorizer:")
    print(e)
    model = None
    vectorizer = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether model loaded
    if model is None or vectorizer is None:
        return render_template(
            "index.html",
            error="Model files could not be loaded. Please check the .pkl files."
        )

    # Get message from HTML form
    message = request.form.get("message", "").strip()

    # Check empty message
    if not message:
        return render_template(
            "index.html",
            error="Please enter a message."
        )

    try:
        # Convert message into Bag-of-Words
        message_vector = vectorizer.transform([message])

        # Predict class
        prediction = model.predict(message_vector)[0]

        # Get probabilities
        probabilities = model.predict_proba(message_vector)[0]

        # Get class names
        classes = model.classes_

        # Create probability dictionary
        probability_data = {}

        for class_name, probability in zip(classes, probabilities):
            probability_data[class_name] = round(
                probability * 100, 2
            )

        # Highest probability
        confidence = round(
            max(probabilities) * 100, 2
        )

        return render_template(
            "index.html",
            message=message,
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_data
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)