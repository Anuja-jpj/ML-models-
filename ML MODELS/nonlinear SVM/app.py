from pathlib import Path
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "banknote_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "banknote_model.pkl not found. Run 'python train_model.py' first."
    )

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            variance = float(request.form["variance"])
            skewness = float(request.form["skewness"])
            curtosis = float(request.form["curtosis"])
            entropy = float(request.form["entropy"])
            features = [[variance, skewness, curtosis, entropy]]
            result = int(model.predict(features)[0])
            prediction = "Genuine Banknote" if result == 0 else "Counterfeit Banknote"
        except (ValueError, KeyError):
            error = "Please enter valid numerical values in all four fields."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
