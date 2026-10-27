
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained Linear SVM model
with open("banknote_model.pkl", "rb") as file:
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

            features = [[
                variance,
                skewness,
                curtosis,
                entropy
            ]]

            result = int(model.predict(features)[0])

            if result == 0:
                prediction = "Genuine Banknote"
            else:
                prediction = "Counterfeit Banknote"

        except ValueError:
            error = "Please enter valid numerical values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
