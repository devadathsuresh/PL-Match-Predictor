# 2025/26 prediction evaluation

Training seasons: 2022/23, 2023/24, 2024/25 (1140 matches)
Evaluation season: 2025/26 (380 matches)
The model predicts each match before kickoff using team form from earlier matches. For later fixtures, earlier completed 2025/26 matches contribute to the form features.

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Most-common baseline | 0.426 | 0.333 | 0.199 |
| Logistic regression | 0.468 | 0.434 | 0.402 |
| XGBoost | 0.450 | 0.426 | 0.349 |

Logistic regression correct predictions: 178 of 380.
XGBoost correct predictions: 171 of 380.

## Per-outcome report

```text
              precision    recall  f1-score   support

    Home win       0.56      0.64      0.59       162
        Draw       0.26      0.09      0.13       104
    Away win       0.41      0.58      0.48       114

    accuracy                           0.47       380
   macro avg       0.41      0.43      0.40       380
weighted avg       0.43      0.47      0.43       380
```

## Confusion matrix

Rows are actual outcomes; columns are predicted outcomes in H, D, A order.

```text
[[103  15  44]
 [ 45   9  50]
 [ 37  11  66]]
```

## XGBoost per-outcome report

```text
              precision    recall  f1-score   support

    Home win       0.54      0.52      0.53       162
        Draw       0.00      0.00      0.00       104
    Away win       0.39      0.75      0.51       114

    accuracy                           0.45       380
   macro avg       0.31      0.43      0.35       380
weighted avg       0.35      0.45      0.38       380
```

## XGBoost confusion matrix

Rows are actual outcomes; columns are predicted outcomes in H, D, A order.

```text
[[85  3 74]
 [44  0 60]
 [28  0 86]]
```

The baseline, logistic regression, and XGBoost prediction CSVs contain each match's actual outcome and prediction. Model CSVs also contain probabilities. Keep the dataset credited to Football-Data.co.uk.