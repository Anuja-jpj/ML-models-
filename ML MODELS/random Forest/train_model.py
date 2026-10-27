from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

def main():
    print("Loading California Housing dataset...")
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.copy()
    y_reg = bunch.target.copy()  # Target is in units of $100,000.

    # Data cleaning: replace infinities, then remove rows with missing values.
    X = X.replace([np.inf, -np.inf], np.nan)
    data = X.copy()
    data["target"] = y_reg
    data = data.dropna()
    X = data.drop(columns="target")
    y_reg = data["target"]

    # EDA outputs
    X.describe().T.to_csv(OUT / "feature_summary.csv")
    pd.DataFrame({"median_house_value_100k": y_reg}).describe().to_csv(OUT / "target_summary.csv")
    plt.figure(figsize=(8, 5))
    y_reg.hist(bins=40)
    plt.title("Distribution of Median House Value")
    plt.xlabel("Target (units of $100,000)")
    plt.ylabel("Number of districts")
    plt.tight_layout()
    plt.savefig(OUT / "target_distribution.png", dpi=160)
    plt.close()

    # Same split for regression and classification.
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # Classification threshold is calculated from training data only.
    threshold = float(yr_train.median())
    y_cls = (y_reg >= threshold).astype(int)
    yc_train, yc_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

    # These datasets have numeric features; tree models do not require scaling.
    # Both algorithms receive identical features and identical train/test rows.
    reg_models = {
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=18
        ),
    }
    cls_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=18
        ),
    }

    reg_rows = []
    for name, model in reg_models.items():
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(yr_test, pred)))
        reg_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(yr_test, pred),
            "RMSE": rmse,
            "R2": r2_score(yr_test, pred)
        })
        joblib.dump(model, MODELS / (name.lower().replace(" ", "_") + "_regressor.joblib"))

    cls_rows = []
    for name, model in cls_models.items():
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        cls_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yc_test, pred),
            "Precision": precision_score(yc_test, pred, zero_division=0),
            "Recall": recall_score(yc_test, pred, zero_division=0),
            "F1-score": f1_score(yc_test, pred, zero_division=0)
        })
        joblib.dump(model, MODELS / (name.lower().replace(" ", "_") + "_classifier.joblib"))

    reg_df = pd.DataFrame(reg_rows)
    cls_df = pd.DataFrame(cls_rows)
    reg_df.to_csv(OUT / "regression_comparison.csv", index=False)
    cls_df.to_csv(OUT / "classification_comparison.csv", index=False)

    # Metric comparison plots
    for metric in ["MAE", "RMSE", "R2"]:
        plt.figure(figsize=(7, 4))
        plt.bar(reg_df["Model"], reg_df[metric])
        plt.title(f"Regression Model Comparison: {metric}")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(OUT / f"regression_{metric.lower()}.png", dpi=160)
        plt.close()

    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        plt.figure(figsize=(7, 4))
        plt.bar(cls_df["Model"], cls_df[metric])
        plt.title(f"Classification Model Comparison: {metric}")
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(OUT / f"classification_{metric.lower().replace('-', '_')}.png", dpi=160)
        plt.close()

    # Confusion matrix for each classifier
    for name, model in cls_models.items():
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X_test, yc_test, display_labels=["Lower price", "Higher price"],
            cmap="Blues", values_format="d"
        )
        disp.ax_.set_title(f"{name} Confusion Matrix")
        disp.figure_.tight_layout()
        disp.figure_.savefig(OUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close(disp.figure_)

    # Feature importance for the Random Forest models
    pd.DataFrame({
        "Feature": X.columns,
        "Regression_Importance": reg_models["Random Forest"].feature_importances_,
        "Classification_Importance": cls_models["Random Forest"].feature_importances_,
    }).sort_values("Regression_Importance", ascending=False).to_csv(
        OUT / "feature_importance.csv", index=False
    )

    metadata = {
        "features": list(X.columns),
        "classification_threshold_100k": threshold,
        "classification_threshold_dollars": threshold * 100000,
        "target_unit": "Target values are in units of $100,000",
        "random_state": RANDOM_STATE,
        "test_size": 0.2
    }
    with open(MODELS / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Actual vs predicted scatter plot for both regression models
    plt.figure(figsize=(6, 6))
    for name, model in reg_models.items():
        plt.scatter(yr_test, model.predict(X_test), s=8, alpha=0.35, label=name)
    lo, hi = float(yr_test.min()), float(yr_test.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual target (units of $100,000)")
    plt.ylabel("Predicted target (units of $100,000)")
    plt.title("Actual vs Predicted House Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
    plt.close()

    print("\nREGRESSION COMPARISON")
    print(reg_df.round(4).to_string(index=False))
    print("\nCLASSIFICATION COMPARISON")
    print(cls_df.round(4).to_string(index=False))
    print(f"\nClassification threshold: ${threshold * 100000:,.0f}")
    print(f"Outputs saved in: {OUT}")
    print(f"Models saved in: {MODELS}")

if __name__ == "__main__":
    main()
