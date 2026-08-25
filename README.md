# ✈️ Flight Delay Prediction & Model Explainability

An end-to-end machine learning project to predict **total arrival delay minutes** for a given airline–airport combination and month.

The project covers exploratory data analysis, preprocessing, regression modeling, cross-validation, ensemble learning, XGBoost, and model explainability using SHAP.

---

## 🎯 Problem Statement

Flight delays are influenced by factors such as traffic volume, airport characteristics, airline operations, and time-related patterns.

The objective of this project is to answer:

> **Given an airline, airport, month, year, number of arriving flights, and peak-delay indicator, how many total arrival-delay minutes can we expect?**

The project also goes beyond prediction and investigates:

> **What factors are driving the model's predictions?**

---

## 📊 Dataset

The dataset contains approximately **398,000 records** representing airline–airport–month observations.

The original dataset contains operational and delay-related variables. During EDA and preprocessing, several highly correlated and potentially target-leaking variables were investigated.

For the final predictive models, the feature set was reduced to:

* `year`
* `month`
* `carrier_name`
* `airport_name`
* `arr_flights`
* `peak_delay`

### Target

* `arr_delay` — total arrival delay minutes

---

## 🔎 Exploratory Data Analysis

EDA was performed to understand:

* Distribution of numerical variables
* Skewness and outliers
* Relationships between numerical variables and the target
* Correlations and multicollinearity
* Missing values
* Categorical variables
* Relationships between flight volume and delays

One important finding was that several operational delay variables were strongly correlated with the target.

Rather than simply adding every available variable to the predictive model, the project investigated whether a more realistic prediction setup could be built using variables that would be available when making a prediction.

### Important EDA Finding

`arr_flights` showed a strong relationship with total arrival delay.

The relationship was not necessarily linear, but higher flight volumes generally corresponded to larger total delay minutes.

---

## ⚙️ Preprocessing

A `ColumnTransformer` and Scikit-learn pipelines were used to keep preprocessing consistent between training and testing.

### Categorical features

* `carrier_name`
* `airport_name`

were encoded using **OneHotEncoder**.

### Numerical features

* `year`
* `month`
* `arr_flights`
* `peak_delay`

were passed through the preprocessing pipeline.

Using pipelines ensured that preprocessing was fitted only on the training data and consistently applied to unseen data.

---

# 🤖 Models

The project compared several regression approaches:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor

Evaluation metrics used:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination
* Cross-validation R²

---

## 📈 Model Results

### Linear Regression

Using the reduced feature set:

| Metric |   Train |       Test |
| ------ | ------: | ---------: |
| MAE    | 2004.79 |    2016.34 |
| RMSE   | 5921.79 |    5822.17 |
| R²     |  0.7855 | **0.7899** |

The relatively similar train and test performance suggested that the linear model was not heavily overfitting, but its performance was limited by the nonlinear nature of the problem.

---

### Decision Tree

Increasing tree depth improved training performance substantially, but deeper trees showed increasing signs of overfitting.

Cross-validation was therefore used to evaluate generalization rather than relying only on the training/test split.

A 5-fold cross-validation experiment produced:

* **Mean CV R²: 0.8253**
* **Standard deviation: 0.0076**

This showed that the model generalized reasonably consistently across folds.

---

### Random Forest

Random Forest substantially improved predictive performance compared with a single decision tree.

For the tested configuration with 100 trees and maximum depth 25:

| Metric |   Train |       Test |
| ------ | ------: | ---------: |
| MAE    |  714.07 |    1185.08 |
| RMSE   | 1817.18 |    4059.85 |
| R²     |  0.9798 | **0.8978** |

The Random Forest also produced:

* **OOB R² ≈ 0.887**
* **Mean 5-fold CV R² ≈ 0.885**

The gap between training and validation performance demonstrated some overfitting, but the validation results remained strong.

---

### XGBoost ⭐

XGBoost provided the strongest test-set performance among the models evaluated.

| Metric |   Train |        Test |
| ------ | ------: | ----------: |
| MAE    | 1157.55 | **1224.32** |
| RMSE   | 3359.49 | **3862.12** |
| R²     |  0.9310 |  **0.9076** |

The relatively small difference between the train and test MAE, together with the strong test R², indicated good generalization for the final model.

---

## 🏆 Model Comparison

| Model             |    Test R² |
| ----------------- | ---------: |
| Linear Regression |     0.7899 |
| Decision Tree     |     ~0.96* |
| Random Forest     |     0.8978 |
| **XGBoost**       | **0.9076** |

*Decision-tree performance varied considerably with maximum depth and was therefore evaluated alongside cross-validation.

### Final Model

**XGBoost Regressor** was selected as the final predictive model because it achieved the strongest test-set performance while maintaining a relatively good balance between training and test performance.

---

# 🧠 Model Explainability with SHAP

Prediction accuracy alone does not explain **why** the model makes a prediction.

SHAP (SHapley Additive exPlanations) was therefore used to investigate how individual features influence the XGBoost model.

Three major SHAP analyses were performed.

---

## 1. SHAP Beeswarm — Global Explanation

The SHAP beeswarm plot was used to understand which features generally have the largest influence on predictions.

A major finding was:

> **`arr_flights` was the strongest feature influencing the model's predictions.**

Airport, carrier, year, month, and peak-delay information also contributed to predictions.

Importantly, SHAP explains **model behavior**, not causal relationships.

---

## 2. SHAP Waterfall — Local Explanation

Waterfall plots were used to explain individual predictions.

For an individual airport–airline–month observation, the plot starts from the model's baseline prediction and shows how individual features push the prediction higher or lower.

For example:

* Low `arr_flights` can substantially reduce the predicted delay.
* `peak_delay` can increase the prediction.
* Year and month can make smaller positive or negative contributions.
* Specific airline and airport categories can shift the prediction.

This allows individual predictions to be explained rather than treating the XGBoost model as a black box.

---

## 3. SHAP Dependence Plot — Feature Behaviour

A SHAP dependence plot was created for `arr_flights`.

The plot showed a strong positive relationship between:

**Number of arriving flights → SHAP contribution to predicted delay**

As `arr_flights` increased, its SHAP contribution generally increased.

This indicates that the model has learned that higher flight volumes are associated with higher total arrival-delay minutes.

The spread of SHAP values at similar flight volumes also suggests that the effect of flight volume depends on other variables such as airport, carrier, and time.

---

# 💡 Key Findings

### 1. Flight volume is highly important

`arr_flights` emerged as one of the strongest predictors.

Higher flight volumes generally resulted in larger predicted total delay minutes.

### 2. Nonlinear models outperform linear regression

Tree-based ensemble methods captured relationships that Linear Regression could not represent effectively.

### 3. A single Decision Tree can overfit

Increasing tree depth substantially improved training performance, but the improvement on unseen data eventually plateaued.

Cross-validation was therefore important for evaluating generalization.

### 4. Ensemble models provide stronger generalization

Random Forest and XGBoost significantly improved test-set performance compared with Linear Regression.

### 5. Prediction ≠ causation

SHAP explains **what the model used to make predictions**.

It does **not** prove that a feature causally produces delays.

For example, the strong influence of `arr_flights` means the model uses flight volume to predict delays; it does not by itself prove that increasing the number of flights causes delays.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Jupyter Notebook

---

# 📁 Project Structure

```text
flight-delay-prediction/
│
├── eda.ipynb
├── preprocessing.ipynb
├── models.ipynb
├── model_explanation.ipynb
└── README.md
```

### Notebooks

**`eda.ipynb`**

Exploratory data analysis, distributions, relationships, correlations, and feature investigation.

**`preprocessing.ipynb`**

Data cleaning, feature preparation, target definition, and preprocessing decisions.

**`models.ipynb`**

Model training, evaluation, hyperparameter experiments, and cross-validation.

**`model_explanation.ipynb`**

XGBoost model explanation using SHAP beeswarm, waterfall, and dependence plots.

---

# 🚀 Future Improvements

Possible extensions to this project include:

* Time-aware validation to better simulate forecasting future months
* Hyperparameter optimization for XGBoost
* Additional feature engineering
* SHAP interaction analysis
* Causal inference to investigate potential causes of delays
* Power BI dashboard for operational visualization
* Deployment of the final model as an API or web application

---

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow, from exploratory analysis and preprocessing to model comparison and explainability.

The final XGBoost model achieved a **test R² of approximately 0.908**, while SHAP analysis provided insight into how variables such as flight volume, airport, carrier, and temporal features influenced individual predictions.

The project therefore goes beyond simply asking:

> **"How accurately can we predict flight delays?"**

and also asks:

> **"What did the model learn, and why did it make this prediction?"**
