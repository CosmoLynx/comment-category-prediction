# Comment Category Prediction

A machine learning project for predicting discussion comment categories from text and metadata using classical NLP, feature engineering, and ensemble learning.

## Overview
This project builds a multi-class classifier that predicts one of four labels (`0` to `3`) for online discussion comments using comment text plus structured metadata such as timestamps, votes, emoticons, flags, and demographic fields. The training set contains **198,000 labeled comments** and the test set contains **102,000 unlabeled comments**. The final selected model is a **stacking ensemble** that achieved a **macro F1-score of about 0.82** on the validation split.

## Problem Statement
Large-scale content platforms generate huge volumes of user comments, making manual review and categorization expensive and slow. The objective of this project is to automate comment category prediction with a model that performs well across all classes, not just the majority one.

A key challenge in this dataset is **class imbalance**:
- Class `0` is the dominant class.
- Class `2` is the second largest.
- Classes `1` and `3` are minority classes.
- Class `3` is especially rare, making macro F1 optimization difficult.

## Dataset
Each row contains a comment and associated metadata. Important fields include:
- `created_date`
- `post_id`
- `emoticon_1`, `emoticon_2`, `emoticon_3`
- `upvote`, `downvote`
- `if_1`, `if_2`
- `race`, `religion`, `gender`
- `disability`
- `comment`
- `label` (train only)

### Data notes
- Around **73%** of `race`, `religion`, and `gender` values are missing.
- There are no fully duplicated rows, but **157 comments** have duplicate text.
- Engagement features are highly right-skewed.
- `if_2` is the strongest structured feature correlated with the target.

## Approach

### 1. Exploratory Data Analysis
EDA focused on:
- Missing values and duplicate checks
- Label imbalance
- Distribution of numeric features
- Comment length patterns
- Temporal activity trends
- Correlation of structured features with label

### 2. Feature Engineering
Engineered features include:
- **Datetime features**: month, day of week, hour, time of day, weekend flag
- **Text statistics**: character count, word count, average word length, uppercase ratio
- **Style features**: punctuation counts, sentence count, elongated word count, unique word ratio
- **Engagement features**: net score, total reactions, vote ratio, reaction intensity
- **Emoticon features**: total emoticons, emoticon density, has-emoticon flag
- **Missing-category handling**: filling `race`, `religion`, and `gender` with `Unknown`
- **Skew reduction**: log transforms on features such as `if_1`, `if_2`, upvotes, downvotes, and reaction counts

### 3. Text Processing
The text pipeline includes:
- HTML entity cleanup
- Whitespace normalization
- TF-IDF vectorization on comment text
- Vocabulary size of **149,597** terms before feature selection
- Chi-square feature selection down to **30,000** text features

### 4. Structured Feature Pipeline
A `ColumnTransformer` was used to preprocess structured data:
- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown='ignore')` for categorical features
- passthrough for binary fields

After concatenating structured and selected text features, the final feature matrix had **30,057 dimensions**.

## Models Tried
The project evaluated multiple baseline and advanced models:
- Logistic Regression
- SGD Classifier
- Passive Aggressive Classifier
- LinearSVC
- LightGBM
- XGBoost
- Stacking Ensemble

## Validation Performance
| Model | Train Macro F1 | Validation Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.886 | 0.805 |
| SGD Classifier | 0.824 | 0.792 |
| Passive Aggressive | 0.872 | 0.792 |
| LinearSVC | 0.904 | 0.806 |
| LightGBM | 0.917 | 0.814 |
| XGBoost | 0.844 | 0.787 |
| **Stacking Ensemble** | **0.898** | **0.820** |

## Final Model
The final solution uses a **StackingClassifier** with the following setup:
- **Base learners**:
  - SGD Classifier
  - Passive Aggressive Classifier
  - LinearSVC
  - LightGBM
  - XGBoost
- **Meta-learner**:
  - LightGBM
- **Cross-validation for stacking**:
  - 5-fold out-of-fold prediction strategy

This combination improved validation macro F1 slightly over the best standalone model and gave the strongest overall performance.

## Key Findings
- **TF-IDF + classical ML** worked very well for this problem.
- Text features carried most of the predictive power.
- LightGBM was the best single model.
- Stacking improved performance through model diversity.
- Minority classes, especially class `3`, remained the hardest to classify.
- Many class `3` samples were confused with class `2`, suggesting overlap in textual patterns.

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- matplotlib
- seaborn
- Jupyter Notebook / Kaggle Notebook

## Repository Structure
```bash
.
├── README.md
├── index.html
├── MLProjectReport.pdf
├── 23f3001400-notebook-t12026.ipynb
└── assets/
```

## How to Run
1. Open the notebook.
2. Install required dependencies.
3. Place the competition `train.csv` and `test.csv` files in the expected input path.
4. Run the notebook cells in sequence.
5. Train the models and generate predictions.
6. Export the final submission CSV.

Example package installation:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

## Results Summary
The project reached a strong validation score with a practical classical NLP pipeline, showing that careful preprocessing, engineered metadata, and ensembling can produce competitive results without transformers.

## Future Improvements
- Fine-tune transformer models such as BERT or RoBERTa
- Apply oversampling methods for minority classes
- Better understand and engineer around `if_2`
- Improve probability calibration
- Explore denser semantic text representations

## Author Notes
This repo documents the end-to-end workflow for a multi-class text classification task: EDA, preprocessing, feature engineering, modeling, tuning, and final submission generation.
