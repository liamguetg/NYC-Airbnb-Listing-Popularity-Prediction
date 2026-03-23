# NYC Airbnb Listing Popularity Prediction

This project applies machine learning to the 2019 New York City Airbnb dataset to predict `reviews_per_month` (used as a proxy for listing popularity).  
It was completed as a full end-to-end regression workflow in CPSC 330, with emphasis on data preprocessing, model selection, and interpretation.

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

- Baseline model established a reference error (RMSE around 1.63).
- Ridge regression improved substantially over baseline after preprocessing and feature engineering.
- Best Ridge configuration (`alpha` selected via CV) achieved:
  - Cross-validated RMSE around **1.26**
  - Test RMSE around **1.17**
  - Test R^2 around **0.48**
- Results suggested moderate predictive power and reasonable generalization without major overfitting.

## What This Project Demonstrates

- Ability to design an end-to-end ML pipeline for mixed-type tabular data.
- Practical handling of real-world data issues: missingness, skew, outliers, and high-cardinality categorical/text features.
- Correct use of validation methodology for model comparison and hyperparameter tuning.
- Understanding of model interpretability techniques beyond raw performance metrics.

## Repository Contents

- `hw5.ipynb` - full notebook with EDA, preprocessing, modeling, evaluation, and interpretation
- `data/AB_NYC_2019.csv` - dataset used for training/evaluation
- `hw5__2_ (1).html` - exported notebook report
