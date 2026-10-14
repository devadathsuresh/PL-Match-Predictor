"""Compare stronger pre-match features using a time-ordered validation split."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "data" / "match_features.csv"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"
VALIDATION_TRAIN_SEASONS = ["2022/23", "2023/24"]
VALIDATION_SEASON = "2024/25"
FINAL_TRAIN_SEASONS = ["2022/23", "2023/24", "2024/25"]
TEST_SEASON = "2025/26"

TEAM_FEATURES = ["HomeTeam", "AwayTeam"]
FORM_FEATURES = [
    "HomePointsLast5",
    "HomeGoalsForLast5",
    "HomeGoalsAgainstLast5",
    "HomeGamesBefore",
    "AwayPointsLast5",
    "AwayGoalsForLast5",
    "AwayGoalsAgainstLast5",
    "AwayGamesBefore",
]
VENUE_FEATURES = [
    "HomeHomePointsLast5",
    "HomeHomeGoalsForLast5",
    "HomeHomeGoalsAgainstLast5",
    "HomeHomeGamesBefore",
    "AwayAwayPointsLast5",
    "AwayAwayGoalsForLast5",
    "AwayAwayGoalsAgainstLast5",
    "AwayAwayGamesBefore",
]
ELO_FEATURES = ["HomeEloBefore", "AwayEloBefore", "EloDifference", "HomeEloExpected"]
ODDS_FEATURES = ["ClosingOddsHome", "ClosingOddsDraw", "ClosingOddsAway"]
LABELS = ["H", "D", "A"]
LABEL_NAMES = ["Home win", "Draw", "Away win"]
LABEL_TO_NUMBER = {"H": 0, "D": 1, "A": 2}
NUMBER_TO_LABEL = {number: label for label, number in LABEL_TO_NUMBER.items()}


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
    }


def markdown_metrics_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.name} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.macro_f1:.3f} |"
        )
    return "\n".join(lines)


def make_pipeline(kind: str, include_odds: bool, include_advanced: bool) -> Pipeline:
    # Use a new list per model; extending FORM_FEATURES in place would duplicate columns.
    numeric_features = FORM_FEATURES.copy()
    if include_advanced:
        numeric_features += VENUE_FEATURES + ELO_FEATURES
    if include_odds:
        numeric_features += ODDS_FEATURES
    numeric_steps = [("fill_missing", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "logistic":
        numeric_steps.append(("scale", StandardScaler()))

    preprocess = ColumnTransformer(
        transformers=[
            ("team_names", OneHotEncoder(handle_unknown="ignore"), TEAM_FEATURES),
            ("numbers", Pipeline(steps=numeric_steps), numeric_features),
        ]
    )

    if kind == "logistic":
        classifier = LogisticRegression(max_iter=3000)
    else:
        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=350,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
    return Pipeline(steps=[("prepare", preprocess), ("classifier", classifier)])


def candidate_features(include_odds: bool, include_advanced: bool) -> list[str]:
    features = TEAM_FEATURES + FORM_FEATURES
    if include_advanced:
        features += VENUE_FEATURES + ELO_FEATURES
    if include_odds:
        features += ODDS_FEATURES
    return features


def train_candidate(
    kind: str, include_odds: bool, include_advanced: bool, train: pd.DataFrame
) -> Pipeline:
    features = candidate_features(include_odds, include_advanced)
    model = make_pipeline(kind, include_odds, include_advanced)
    labels = train["FTR"]
    if kind == "xgboost":
        labels = labels.map(LABEL_TO_NUMBER)
    model.fit(train[features], labels)
    return model


def predict(
    model: Pipeline, kind: str, include_odds: bool, include_advanced: bool, frame: pd.DataFrame
) -> tuple[pd.Series, np.ndarray]:
    features = candidate_features(include_odds, include_advanced)
    raw_predictions = model.predict(frame[features])
    probabilities = model.predict_proba(frame[features])
    if kind == "xgboost":
        labels = pd.Series([NUMBER_TO_LABEL[int(value)] for value in raw_predictions], index=frame.index)
        class_labels = [NUMBER_TO_LABEL[int(value)] for value in model.named_steps["classifier"].classes_]
    else:
        labels = pd.Series(raw_predictions, index=frame.index)
        class_labels = list(model.classes_)

    # Reorder probability columns into the familiar home/draw/away order.
    probability_order = [class_labels.index(label) for label in LABELS]
    return labels, probabilities[:, probability_order]


def save_predictions(frame: pd.DataFrame, predicted: pd.Series, probabilities: np.ndarray, file_path: Path) -> pd.DataFrame:
    actual_names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    results = frame[["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    results = results.rename(columns={"FTR": "ActualResult"})
    results["PredictedResult"] = predicted
    results["Correct"] = results["ActualResult"] == results["PredictedResult"]
    results["ActualOutcome"] = results["ActualResult"].map(actual_names)
    results["PredictedOutcome"] = results["PredictedResult"].map(actual_names)
    for index, column in enumerate(["ProbHomeWin", "ProbDraw", "ProbAwayWin"]):
        results[column] = probabilities[:, index]
    results.to_csv(file_path, index=False)
    return results


def main() -> None:
    if not DATA_FILE.exists():
        print("Feature table not found. Run build_features.py first.")
        return

    matches = pd.read_csv(DATA_FILE)
    required = (
        TEAM_FEATURES
        + FORM_FEATURES
        + VENUE_FEATURES
        + ELO_FEATURES
        + ODDS_FEATURES
        + ["Season", "Date", "FTR"]
    )
    missing = set(required) - set(matches.columns)
    if missing:
        print(f"Feature table is missing columns: {sorted(missing)}. Run build_features.py again.")
        return

    validation_train = matches[matches["Season"].isin(VALIDATION_TRAIN_SEASONS)].copy()
    validation = matches[matches["Season"] == VALIDATION_SEASON].copy()
    final_train = matches[matches["Season"].isin(FINAL_TRAIN_SEASONS)].copy()
    test = matches[matches["Season"] == TEST_SEASON].copy()
    if validation.empty or test.empty or validation_train.empty or final_train.empty:
        print("Required training, validation, or 2025/26 evaluation seasons are missing.")
        return

    # Choose features and model family on 2024/25; do not use 2025/26 to choose.
    candidates = [
        ("Logistic regression: team form", "logistic", False, False),
        ("XGBoost: team form", "xgboost", False, False),
        ("Logistic regression: form + closing odds", "logistic", True, False),
        ("XGBoost: form + closing odds", "xgboost", True, False),
        ("Logistic regression: form + venue + Elo", "logistic", False, True),
        ("XGBoost: form + venue + Elo", "xgboost", False, True),
        ("Logistic regression: all features", "logistic", True, True),
        ("XGBoost: all features", "xgboost", True, True),
    ]
    validation_rows = []
    candidate_models = {}
    for name, kind, include_odds, include_advanced in candidates:
        candidate_model = train_candidate(kind, include_odds, include_advanced, validation_train)
        candidate_predictions, _ = predict(
            candidate_model, kind, include_odds, include_advanced, validation
        )
        candidate_metrics = metrics(validation["FTR"], candidate_predictions)
        validation_rows.append({"name": name, **candidate_metrics})
        candidate_models[name] = (kind, include_odds, include_advanced)

    validation_baseline = DummyClassifier(strategy="most_frequent")
    validation_baseline.fit(validation_train[TEAM_FEATURES + FORM_FEATURES], validation_train["FTR"])
    validation_base_predictions = validation_baseline.predict(validation[TEAM_FEATURES + FORM_FEATURES])
    validation_rows.append(
        {"name": "Most-common baseline", **metrics(validation["FTR"], validation_base_predictions)}
    )

    market_choices = np.argmin(validation[ODDS_FEATURES].to_numpy(dtype=float), axis=1)
    market_predictions = pd.Series([LABELS[index] for index in market_choices], index=validation.index)
    validation_rows.append(
        {"name": "Closing-odds market pick", **metrics(validation["FTR"], market_predictions)}
    )

    validation_scores = pd.DataFrame(validation_rows).sort_values(
        ["accuracy", "macro_f1"], ascending=False, kind="stable"
    )
    selected_name = validation_scores[
        validation_scores["name"].isin(candidate_models)
    ].iloc[0]["name"]
    selected_kind, selected_has_odds, selected_has_advanced = candidate_models[selected_name]
    print("2024/25 validation: trained on 2022/23–2023/24")
    print(validation_scores.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "balanced_accuracy": "{:.3f}".format,
        "macro_f1": "{:.3f}".format,
    }))
    print(f"\nSelected using validation accuracy: {selected_name}")

    # Refit the validation-selected model on all three requested training seasons.
    final_model = train_candidate(
        selected_kind, selected_has_odds, selected_has_advanced, final_train
    )
    final_predictions, final_probabilities = predict(
        final_model, selected_kind, selected_has_odds, selected_has_advanced, test
    )
    final_metrics = metrics(test["FTR"], final_predictions)

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_features = TEAM_FEATURES + FORM_FEATURES
    majority_model.fit(final_train[majority_features], final_train["FTR"])
    majority_predictions = majority_model.predict(test[majority_features])
    majority_metrics = metrics(test["FTR"], pd.Series(majority_predictions, index=test.index))

    market_choices = np.argmin(test[ODDS_FEATURES].to_numpy(dtype=float), axis=1)
    market_predictions = pd.Series([LABELS[index] for index in market_choices], index=test.index)
    market_metrics = metrics(test["FTR"], market_predictions)

    REPORT_DIR.mkdir(exist_ok=True)
    prediction_file = REPORT_DIR / "improved_predictions_2025_2026.csv"
    results = save_predictions(test, final_predictions, final_probabilities, prediction_file)
    selected_model_metrics = metrics(test["FTR"], results["PredictedResult"])

    # Compare with the original form-only runs, which used the same train/test seasons.
    old_logistic = pd.read_csv(REPORT_DIR / "predictions_2025_2026.csv")
    old_xgboost = pd.read_csv(REPORT_DIR / "predictions_2025_2026_xgboost.csv")
    original_metrics = {
        "Original logistic regression (form only)": metrics(old_logistic["ActualResult"], old_logistic["PredictedResult"]),
        "Original XGBoost (form only)": metrics(old_xgboost["ActualResult"], old_xgboost["PredictedResult"]),
    }

    score_rows = [
        {"name": "Most-common baseline", **majority_metrics},
        {"name": "Closing-odds market pick", **market_metrics},
        *[{"name": name, **score} for name, score in original_metrics.items()],
        {"name": f"Selected: {selected_name}", **selected_model_metrics},
    ]
    score_table = pd.DataFrame(score_rows)
    score_table = score_table.sort_values(["accuracy", "macro_f1"], ascending=False, kind="stable")

    report = [
        "# Accuracy improvement experiment",
        "",
        "## Method",
        "",
        "Candidate model and feature choices were selected on 2024/25, trained on 2022/23–2023/24. "
        "The selected setup was then refit on 2022/23–2024/25 and evaluated on 2025/26.",
        "Expanded team form includes each team's venue-specific last-five record and a running Elo "
        "strength rating. The odds-assisted model uses average closing home/draw/away odds. These are pre-match market "
        "information; a high score partly reflects the market's knowledge, not only the model's own "
        "team-form analysis. The closing-odds pick is included as a transparent market benchmark.",
        "",
        "## Validation results (2024/25)",
        "",
        markdown_metrics_table(validation_scores),
        "",
        f"Selected by validation accuracy: **{selected_name}**.",
        "",
        "## Final 2025/26 evaluation",
        "",
        markdown_metrics_table(score_table),
        "",
        f"The selected model got **{int(results['Correct'].sum())} of {len(results)}** matches correct.",
        "",
        "### Selected model per-outcome results",
        "",
        "```text",
        classification_report(
            test["FTR"], final_predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0
        ).rstrip(),
        "```",
        "",
        "The match-by-match file includes actual and predicted outcomes, correctness, and class probabilities.",
    ]
    (REPORT_DIR / "accuracy_improvement_2025_2026.md").write_text("\n".join(report), encoding="utf-8")
    model_file = MODEL_DIR / "improved_model.joblib"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, model_file)

    print(f"\n2025/26 selected model accuracy: {selected_model_metrics['accuracy']:.3f}")
    print(f"Correct predictions: {int(results['Correct'].sum())} of {len(results)}")
    print(f"Saved comparison to {REPORT_DIR / 'accuracy_improvement_2025_2026.md'}")
    print(f"Saved match outcomes to {prediction_file}")
    print(f"Saved trained model to {model_file}")


if __name__ == "__main__":
    main()
