# Heart Disease Prediction using Machine Learning & Streamlit

---

## a) Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and prediction of heart disease can help healthcare professionals take preventive measures and improve patient outcomes.

The objective of this project is to build and compare multiple Machine Learning classification models to predict whether a person is likely to have heart disease based on health indicators.  
An interactive Streamlit web application is also developed to demonstrate real-time prediction and model comparison.

---

## b) Dataset Description

Dataset: **CDC Heart Disease Indicators Dataset**

This dataset contains health-related survey responses collected by the CDC (Behavioral Risk Factor Surveillance System).

Key characteristics:
- Total records: 300,000+ (subset of 20,000 used for training)
- Features: 17 health and lifestyle attributes
- Target variable: **HeartDisease**
  - Yes → Person has heart disease
  - No → Person does not have heart disease

Example Features:
- BMI – Body Mass Index
- Smoking – Smoking habit
- AlcoholDrinking – Alcohol consumption
- Stroke – History of stroke
- PhysicalHealth – Number of days physical health not good
- MentalHealth – Number of days mental health not good
- DiffWalking – Difficulty walking
- AgeCategory – Age group
- Diabetic – Diabetes status
- SleepTime – Average sleep hours

The dataset contains both numerical and categorical features, which were encoded before training.

---

## c) Models Used and Evaluation Metrics

The following Machine Learning models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Evaluation Metrics Used:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

<!-- METRICS_TABLE_START -->

| Model               |   Accuracy |   AUC |   Precision |   Recall |    F1 |   MCC |
|:--------------------|-----------:|------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.91  | 0.813 |       0.531 |    0.094 | 0.16  | 0.196 |
| Decision Tree       |      0.853 | 0.584 |       0.225 |    0.257 | 0.24  | 0.159 |
| KNN                 |      0.901 | 0.681 |       0.34  |    0.099 | 0.154 | 0.143 |
| Naive Bayes         |      0.84  | 0.784 |       0.275 |    0.47  | 0.347 | 0.275 |
| Random Forest       |      0.906 | 0.785 |       0.387 |    0.066 | 0.113 | 0.13  |
| XGBoost             |      0.904 | 0.795 |       0.398 |    0.13  | 0.196 | 0.187 |

<!-- METRICS_TABLE_END -->

---

## Model Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Provided strong baseline performance and handled large dataset efficiently. |
| Decision Tree | Easy to interpret but showed slight overfitting compared to ensemble models. |
| KNN | Performed reasonably well but training and prediction time increased with dataset size. |
| Naive Bayes | Fastest model; however, independence assumption limits performance. |
| Random Forest | Improved performance significantly by reducing overfitting using bagging. |
| XGBoost | Achieved best overall performance due to boosting and handling complex feature interactions. |

