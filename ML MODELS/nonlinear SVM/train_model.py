"""Train an RBF-kernel (non-linear) SVM on the UCI Banknote Authentication dataset."""
from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
COLUMNS = ["variance", "skewness", "curtosis", "entropy", "class"]

print("Downloading the UCI banknote dataset...")
data = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
X = data[["variance", "skewness", "curtosis", "entropy"]]
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RBF kernel allows a non-linear decision boundary.
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

with (BASE_DIR / "banknote_model.pkl").open("wb") as file:
    pickle.dump(model, file)
print("Saved model to banknote_model.pkl")
