# NYC Airbnb Listing Popularity Prediction

This project applies machine learning to the 2019 New York City Airbnb dataset to predict `reviews_per_month` (used as a proxy for listing popularity).  
It was completed as a full end-to-end regression workflow, with emphasis on data preprocessing, model selection, and interpretation.

## Project Objective

- Build a regression model that estimates listing popularity before new listings are posted.
- Compare baseline and learned models using cross-validated metrics.
- Interpret which listing characteristics most strongly influence predictions.

## Dataset

- **Source:** Kaggle - New York City Airbnb Open Data (`AB_NYC_2019.csv`)
- **Task type:** Supervised regression
- **Target variable:** `reviews_per_month`
- **Train/test strategy:** Holdout split with cross-validation during model development

## Technologies and Tools

- **Language:** Python
- **Core libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **ML stack:** `scikit-learn`
  - `Pipeline`, `ColumnTransformer`
  - `SimpleImputer`, `StandardScaler`, `OneHotEncoder`
  - `CountVectorizer` (for listing title text)
  - `KBinsDiscretizer`, `PowerTransformer`
  - `DummyRegressor`, `Ridge`, `RandomForestRegressor`, `SVR`
  - `GridSearchCV`, `cross_validate`
- **Hyperparameter search:** `optuna`
- **Model explainability:** `shap`

## Data Science and ML Methods Used

- **Exploratory Data Analysis (EDA):**
  - Distribution analysis (histograms)
  - Outlier analysis (box plots, IQR-based reasoning)
  - Summary statistics for numerical variables
- **Feature engineering:**
  - Converted `last_review` into time-based recency (`days_since_last_review`)
  - Created aggregate neighborhood/host review-rate style features
  - Incorporated text information from listing names with bag-of-words + n-grams
- **Preprocessing by feature type:**
  - Median and constant imputation for missing values
  - Log/power-style transformations for skewed numeric data
  - Standardization for continuous variables
  - One-hot encoding for categorical variables
  - Binning for selected geographic features
- **Modeling and evaluation:**
  - Baseline with `DummyRegressor` (mean predictor)
  - Tuned linear model (`Ridge`) with 5-fold cross-validation and `GridSearchCV`
  - Additional non-linear model experimentation (Random Forest, SVR)
  - Primary metric: **RMSE** (with supporting **R^2**)
- **Interpretation:**
  - SHAP analysis on a tree-based model to inspect influential features

## Key Results

- **Baseline (`DummyRegressor`):** Large test error (RMSE ~**1.56**) and **negative R²**, setting a low bar for any serious model.
- **Ridge (tuned with `GridSearchCV`):** Strong linear baseline on CV — mean CV RMSE ~**1.07** and R² ~**0.42** — but outperformed by nonlinear methods on the same pipeline.
- **Final model — tuned Gradient Boosting Regressor (`sklearn`):** Chosen after comparing RF, GBR, k-NN, and SVR on CV and refining GBR with **Optuna** (best CV RMSE ~**0.847** on training folds). Refit on full training data and evaluated once on the **30% holdout test set**:
  - **Test RMSE ~0.90** (≈ **0.895**)
  - **Test MAE ~0.46**
  - **Test R² ~0.67** (≈ **0.669**)
- Test error is close to the tuned CV estimate, consistent with **good generalization**. **SHAP** analysis highlights **`number_of_reviews`** and **`days_since_last_review`** as the main drivers of predicted activity.

## What This Project Demonstrates

- Ability to design an end-to-end ML pipeline for mixed-type tabular data.
- Practical handling of real-world data issues: missingness, skew, outliers, and high-cardinality categorical/text features.
- Correct use of validation methodology for model comparison and hyperparameter tuning.
- Understanding of model interpretability techniques beyond raw performance metrics.

## Repository Contents

- `fullAnalysis.ipynb` — full notebook (EDA, preprocessing, modeling, Optuna, test evaluation, SHAP)
- `requirements.txt` — Python dependencies (`pip install -r requirements.txt`)
- `data/AB_NYC_2019.csv` — dataset used for training/evaluation
