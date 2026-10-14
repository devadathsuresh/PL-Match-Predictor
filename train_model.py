"""Train and evaluate a first win/draw/loss classifier."""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "data" / "match_features.csv"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "logistic_regression.joblib"
TEST_SEASON = "2025/26"
TRAIN_SEASONS = ["2022/23", "2023/24", "2024/25"]

CATEGORICAL_FEATURES = ["HomeTeam", "AwayTeam"]
NUMERIC_FEATURES = [
    "HomePointsLast5",
    "HomeGoalsForLast5",
    "HomeGoalsAgainstLast5",
    "HomeGamesBefore",
    "AwayPointsLast5",
    "AwayGoalsForLast5",
    "AwayGoalsAgainstLast5",
    "AwayGamesBefore",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LABELS = ["H", "D", "A"]
LABEL_NAMES = ["Home win", "Draw", "Away win"]


def make_model() -> Pipeline:
    """Prepare mixed data columns, then fit logistic regression."""
    prepare_numbers = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    prepare_columns = ColumnTransformer(
        transformers=[
            ("team_names", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", prepare_numbers, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("prepare", prepare_columns),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def make_xgboost_model() -> Pipeline:
    """Prepare the same inputs, then fit a small XGBoost classifier."""
    prepare_columns = ColumnTransformer(
        transformers=[
            ("team_names", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", SimpleImputer(strategy="median", add_indicator=True), NUMERIC_FEATURES),
        ]
    )
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    return Pipeline(steps=[("prepare", prepare_columns), ("classifier", classifier)])


def score_model(name: str, actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
    }
    print(f"\n{name}")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Macro F1:          {metrics['macro_f1']:.3f}")
    return metrics


def main() -> None:
    if not DATA_FILE.exists():
        print("Feature table not found. Run build_features.py first.")
        return

    matches = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + ["Season", "FTR"]
    missing_columns = set(required_columns) - set(matches.columns)
    if missing_columns:
        print(f"Feature table is missing columns: {sorted(missing_columns)}")
        return

    train = matches[matches["Season"].isin(TRAIN_SEASONS)].copy()
    test = matches[matches["Season"] == TEST_SEASON].copy()
    missing_train_seasons = set(TRAIN_SEASONS) - set(train["Season"].unique())
    if train.empty or test.empty or missing_train_seasons:
        print(
            f"Expected training seasons {TRAIN_SEASONS} and a {TEST_SEASON} test season. "
            f"Missing training seasons: {sorted(missing_train_seasons)}"
        )
        return

    print(f"Training seasons: {', '.join(TRAIN_SEASONS)} ({len(train)} matches)")
    print(f"Evaluation season: {TEST_SEASON} ({len(test)} matches)")

    x_train, y_train = train[FEATURES], train["FTR"]
    x_test, y_test = test[FEATURES], test["FTR"]

    # This baseline always predicts the most common training result.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    baseline_metrics = score_model(
        "Simple baseline (always predict the most common training result)", y_test, baseline_predictions
    )
    outcome_names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    baseline_results = test[["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    baseline_results = baseline_results.rename(columns={"FTR": "ActualResult"})
    baseline_results["PredictedResult"] = baseline_predictions
    baseline_results["Correct"] = baseline_results["ActualResult"] == baseline_results["PredictedResult"]
    baseline_results["ActualOutcome"] = baseline_results["ActualResult"].map(outcome_names)
    baseline_results["PredictedOutcome"] = baseline_results["PredictedResult"].map(outcome_names)
    REPORT_DIR.mkdir(exist_ok=True)
    baseline_results.to_csv(REPORT_DIR / "predictions_2025_2026_baseline.csv", index=False)

    model = make_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    model_metrics = score_model("Logistic regression", y_test, predictions)

    print("\nPer-outcome results:")
    print(classification_report(y_test, predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted; order: H, D, A):")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    # Save each held-out match and logistic regression's probability for each outcome.
    probability_column = {label: index for index, label in enumerate(model.classes_)}
    results = test[["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    results = results.rename(columns={"FTR": "ActualResult"})
    results["PredictedResult"] = predictions
    results["Correct"] = results["ActualResult"] == results["PredictedResult"]
    results["ActualOutcome"] = results["ActualResult"].map(outcome_names)
    results["PredictedOutcome"] = results["PredictedResult"].map(outcome_names)
    for label, column in zip(LABELS, ["ProbHomeWin", "ProbDraw", "ProbAwayWin"]):
        results[column] = probabilities[:, probability_column[label]]

    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "predictions_2025_2026.csv", index=False)
    correct_count = int(results["Correct"].sum())
    print(f"\nCorrect match predictions: {correct_count} of {len(results)}")
    print(f"Saved match-by-match comparison to {REPORT_DIR / 'predictions_2025_2026.csv'}")

    # XGBoost uses the same features and split. Encode the H/D/A target as 0/1/2.
    label_to_number = {"H": 0, "D": 1, "A": 2}
    number_to_label = {number: label for label, number in label_to_number.items()}
    xgb_model = make_xgboost_model()
    xgb_model.fit(x_train, y_train.map(label_to_number))
    xgb_number_predictions = xgb_model.predict(x_test).astype(int)
    xgb_predictions = pd.Series(
        [number_to_label[number] for number in xgb_number_predictions], index=y_test.index
    )
    xgb_probabilities = xgb_model.predict_proba(x_test)
    xgb_metrics = score_model("XGBoost", y_test, xgb_predictions)
    print("XGBoost per-outcome results:")
    print(
        classification_report(
            y_test, xgb_predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0
        )
    )
    print("XGBoost confusion matrix (rows = actual, columns = predicted; order: H, D, A):")
    print(confusion_matrix(y_test, xgb_predictions, labels=LABELS))

    xgb_probability_column = {
        number_to_label[int(number)]: index
        for index, number in enumerate(xgb_model.named_steps["classifier"].classes_)
    }
    xgb_results = test[["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    xgb_results = xgb_results.rename(columns={"FTR": "ActualResult"})
    xgb_results["PredictedResult"] = xgb_predictions
    xgb_results["Correct"] = xgb_results["ActualResult"] == xgb_results["PredictedResult"]
    xgb_results["ActualOutcome"] = xgb_results["ActualResult"].map(outcome_names)
    xgb_results["PredictedOutcome"] = xgb_results["PredictedResult"].map(outcome_names)
    for label, column in zip(LABELS, ["ProbHomeWin", "ProbDraw", "ProbAwayWin"]):
        xgb_results[column] = xgb_probabilities[:, xgb_probability_column[label]]
    xgb_results.to_csv(REPORT_DIR / "predictions_2025_2026_xgboost.csv", index=False)
    print(
        f"XGBoost correct match predictions: {int(xgb_results['Correct'].sum())} "
        f"of {len(xgb_results)}"
    )

    summary = [
        "# 2025/26 prediction evaluation",
        "",
        f"Training seasons: {', '.join(TRAIN_SEASONS)} ({len(train)} matches)",
        f"Evaluation season: {TEST_SEASON} ({len(test)} matches)",
        "The model predicts each match before kickoff using team form from earlier matches. "
        "For later fixtures, earlier completed 2025/26 matches contribute to the form features.",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| Most-common baseline | {baseline_metrics['accuracy']:.3f} | "
        f"{baseline_metrics['balanced_accuracy']:.3f} | {baseline_metrics['macro_f1']:.3f} |",
        f"| Logistic regression | {model_metrics['accuracy']:.3f} | "
        f"{model_metrics['balanced_accuracy']:.3f} | {model_metrics['macro_f1']:.3f} |",
        f"| XGBoost | {xgb_metrics['accuracy']:.3f} | "
        f"{xgb_metrics['balanced_accuracy']:.3f} | {xgb_metrics['macro_f1']:.3f} |",
        "",
        f"Logistic regression correct predictions: {correct_count} of {len(results)}.",
        f"XGBoost correct predictions: {int(xgb_results['Correct'].sum())} of {len(xgb_results)}.",
        "",
        "## Per-outcome report",
        "",
        "```text",
        classification_report(
            y_test, predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0
        ).rstrip(),
        "```",
        "",
        "## Confusion matrix",
        "",
        "Rows are actual outcomes; columns are predicted outcomes in H, D, A order.",
        "",
        "```text",
        str(confusion_matrix(y_test, predictions, labels=LABELS)),
        "```",
        "",
        "## XGBoost per-outcome report",
        "",
        "```text",
        classification_report(
            y_test, xgb_predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0
        ).rstrip(),
        "```",
        "",
        "## XGBoost confusion matrix",
        "",
        "Rows are actual outcomes; columns are predicted outcomes in H, D, A order.",
        "",
        "```text",
        str(confusion_matrix(y_test, xgb_predictions, labels=LABELS)),
        "```",
        "",
        "The baseline, logistic regression, and XGBoost prediction CSVs contain each match's actual "
        "outcome and prediction. Model CSVs also contain probabilities. Keep the dataset credited to "
        "Football-Data.co.uk.",
    ]
    (REPORT_DIR / "evaluation_2025_2026.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved evaluation summary to {REPORT_DIR / 'evaluation_2025_2026.md'}")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    xgb_model_file = MODEL_DIR / "xgboost.joblib"
    joblib.dump(xgb_model, xgb_model_file)
    print(f"Saved trained logistic regression model to {MODEL_FILE}")
    print(f"Saved trained XGBoost model to {xgb_model_file}")


if __name__ == "__main__":
    main()
