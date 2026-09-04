import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Placement.csv")

# Input and output
X = data[["cgpa"]]
y = data["package"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("SLRModel.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")