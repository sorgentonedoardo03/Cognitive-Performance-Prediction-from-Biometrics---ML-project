# Cognitive Performance Prediction from Biometrics and Nutrition

A Machine Learning project exploring the relationship between biometric signals, nutritional habits, and cognitive performance.

Developed at **Politecnico di Milano** — Final Grade: **28/30**.

## Overview

The project aims to predict three dimensions of cognitive performance — **memory, reasoning, and verbal ability** — using wearable-derived biometric data, nutritional information, and demographic features.

The original continuous cognitive scores are transformed into three classes (low, medium, high), turning the task into a supervised classification problem.

The project consists of two main components:

- **Machine Learning Pipeline:** data integration, preprocessing, feature engineering, feature selection, hyperparameter optimization, and model evaluation.
- **Explainability Pipeline:** model interpretation through SHAP, Accumulated Local Effects (ALE), and DiCE counterfactual analysis.

## Machine Learning Pipeline

The pipeline includes:

- Integration of wearable/cognitive and nutritional datasets.
- Exploratory Data Analysis and data cleaning.
- Subject-aware feature scaling and missing-value handling.
- Time-series feature extraction from heart rate and respiration rate signals.
- Target discretization using empirical quantiles.
- Iterative feature selection: Forward, Backward, and Stepwise.
- Hyperparameter optimization using Optuna.
- Model comparison using Leave-One-Subject-Out (LOSO) cross-validation.

### Models

Five classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Classifier (SVC)

### Best Model Results

| Cognitive Target | Best Model | Feature Selection | Features | LOSO Accuracy |
|---|---|---|---:|---:|
| Memory | LightGBM | Forward | 2 | 63.9% |
| Reasoning | SVC | Forward | 8 | 71.8% |
| Verbal | SVC | Stepwise | 6 | 66.4% |

## Explainability

The selected models were further analyzed to understand the contribution and behavior of their input features.

- **SHAP:** feature importance and individual prediction contributions.
- **ALE:** analysis of feature effects across the observed data distribution.
- **DiCE:** generation of counterfactual examples to explore changes associated with alternative predictions.

## Repository Structure

```text
.
├── Explainability/
├── ML_pipeline.ipynb
├── REPORT.pdf
├── pitch_deck.pdf
└── README.md
```

- `ML_pipeline.ipynb`: Machine Learning pipeline and model evaluation.
- `Explainability/`: explainability analysis and generated results.
- `REPORT.pdf`: full technical report.
- `pitch_deck.pdf`: project presentation.

## Limitations

The dataset was relatively small, containing 150 initial observations from 15 participants, reduced to 84 observations after data cleaning.

Although LOSO cross-validation was used to reduce subject-level data leakage, the limited number of participants means that the results should be considered preliminary. The identified relationships are associations, not evidence of causation.

## Future Work

- Validate the findings on larger datasets and more participants.
- Improve and automate the training and explainability pipelines.
- Expand counterfactual analysis across all selected models.
- Explore personalized recommendations based on lifestyle and biometric patterns.
