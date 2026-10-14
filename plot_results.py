"""Create a portfolio chart comparing 2025/26 results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


PROJECT_DIR = Path(__file__).parent
REPORT_DIR = PROJECT_DIR / "reports"
LABELS = ["H", "D", "A"]
LABEL_NAMES = ["Home win", "Draw", "Away win"]
PREDICTION_FILES = {
    "Most-common baseline": REPORT_DIR / "predictions_2025_2026_baseline.csv",
    "Logistic regression": REPORT_DIR / "predictions_2025_2026.csv",
    "XGBoost": REPORT_DIR / "predictions_2025_2026_xgboost.csv",
}


def main() -> None:
    predictions = {}
    for name, file_path in PREDICTION_FILES.items():
        if not file_path.exists():
            print(f"Missing {file_path.name}. Run train_model.py first.")
            return
        predictions[name] = pd.read_csv(file_path)

    metrics = {"Accuracy": [], "Balanced accuracy": [], "Macro F1": []}
    for frame in predictions.values():
        actual = frame["ActualResult"]
        predicted = frame["PredictedResult"]
        metrics["Accuracy"].append(accuracy_score(actual, predicted))
        metrics["Balanced accuracy"].append(balanced_accuracy_score(actual, predicted))
        metrics["Macro F1"].append(
            f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    model_names = list(predictions)
    metric_names = list(metrics)
    x = np.arange(len(metric_names))
    bar_width = 0.24
    for index, name in enumerate(model_names):
        offset = (index - (len(model_names) - 1) / 2) * bar_width
        values = [metrics[metric][index] for metric in metric_names]
        bars = axes[0].bar(x + offset, values, bar_width, label=name)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_title("2025/26 evaluation metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 0.65)
    axes[0].set_xticks(x, metric_names, rotation=15, ha="right")
    axes[0].legend(fontsize=8)

    for axis, name in zip(axes[1:], ["Logistic regression", "XGBoost"]):
        frame = predictions[name]
        matrix = confusion_matrix(frame["ActualResult"], frame["PredictedResult"], labels=LABELS)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{name}: actual vs predicted")
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Actual outcome")
        axis.set_xticks(range(3), LABEL_NAMES, rotation=20, ha="right")
        axis.set_yticks(range(3), LABEL_NAMES)
        midpoint = matrix.max() / 2 if matrix.size else 0
        for row in range(3):
            for column in range(3):
                color = "white" if matrix[row, column] > midpoint else "black"
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Premier League match outcome prediction", fontsize=15)
    output_file = REPORT_DIR / "model_comparison_2025_2026.png"
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved chart to {output_file}")


if __name__ == "__main__":
    main()
