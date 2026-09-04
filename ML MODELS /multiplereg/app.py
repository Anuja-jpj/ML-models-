from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("REPPModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get input values from HTML form
    house_age = float(request.form['house_age'])

    distance = float(request.form['distance'])

    convenience_stores = float(
        request.form['convenience_stores']
    )

    latitude = float(request.form['latitude'])

    longitude = float(request.form['longitude'])


    # Arrange inputs in EXACTLY the same order
    # used while training the model
    input_data = np.array([[
        house_age,
        distance,
        convenience_stores,
        latitude,
        longitude
    ]])


    # Predict house price
    prediction = model.predict(input_data)


    # Display prediction
    prediction_text = (
        f"Predicted House Price : "
        f"{prediction[0]:.2f} per unit area"
    )


    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)