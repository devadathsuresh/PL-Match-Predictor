# Accuracy improvement experiment

## Method

Candidate model and feature choices were selected on 2024/25, trained on 2022/23–2023/24. The selected setup was then refit on 2022/23–2024/25 and evaluated on 2025/26.
Expanded team form includes each team's venue-specific last-five record and a running Elo strength rating. The odds-assisted model uses average closing home/draw/away odds. These are pre-match market information; a high score partly reflects the market's knowledge, not only the model's own team-form analysis. The closing-odds pick is included as a transparent market benchmark.

## Validation results (2024/25)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| XGBoost: all features | 0.563 | 0.496 | 0.441 |
| Closing-odds market pick | 0.555 | 0.483 | 0.417 |
| XGBoost: form + closing odds | 0.550 | 0.487 | 0.438 |
| XGBoost: form + venue + Elo | 0.534 | 0.470 | 0.410 |
| Logistic regression: form + closing odds | 0.521 | 0.460 | 0.416 |
| Logistic regression: all features | 0.518 | 0.453 | 0.404 |
| Logistic regression: form + venue + Elo | 0.511 | 0.446 | 0.401 |
| Logistic regression: team form | 0.500 | 0.441 | 0.397 |
| XGBoost: team form | 0.492 | 0.434 | 0.374 |
| Most-common baseline | 0.408 | 0.333 | 0.193 |

Selected by validation accuracy: **XGBoost: all features**.

## Final 2025/26 evaluation

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Closing-odds market pick | 0.495 | 0.441 | 0.371 |
| Selected: XGBoost: all features | 0.474 | 0.438 | 0.388 |
| Original logistic regression (form only) | 0.468 | 0.434 | 0.402 |
| Original XGBoost (form only) | 0.450 | 0.426 | 0.349 |
| Most-common baseline | 0.426 | 0.333 | 0.199 |

The selected model got **180 of 380** matches correct.

### Selected model per-outcome results

```text
              precision    recall  f1-score   support

    Home win       0.54      0.64      0.59       162
        Draw       0.45      0.05      0.09       104
    Away win       0.40      0.63      0.49       114

    accuracy                           0.47       380
   macro avg       0.47      0.44      0.39       380
weighted avg       0.48      0.47      0.42       380
```

The match-by-match file includes actual and predicted outcomes, correctness, and class probabilities.