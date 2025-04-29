## Cerebral Stroke Prediction Using Machine Learning

This repository contains the full code, data, and analysis for the project *"Cerebral Stroke Prediction Using Machine Learning on an Imbalanced Dataset"*, conducted as part of my MSc Research.  
The goal of the project was to develop a predictive framework for stroke risk detection using ensemble models while specifically addressing extreme class imbalance through advanced resampling techniques.

---

## Folder Structure

```plaintext
├── EDA_of_cerebral_stroke_prediction.ipynb    # Exploratory Data Analysis (EDA) notebook
├── cerebral_stroke_prediction.ipynb           # Main modeling and prediction notebook
├── data/
│   ├── clean_data.csv                          # Cleaned dataset after preprocessing
│   ├── clean_data_encoded.csv                  # Encoded version of the cleaned dataset
│   ├── dataset.csv                             # Original dataset (raw)
│   └── dataset.csv.zip                         # Zipped version of the original dataset
├── images/
│   └── [Plots and visualizations generated from notebooks]
├── README.md                                   # Project documentation
```

---

## Project Overview

The project follows a structured machine learning pipeline:
- Data cleaning and preprocessing.
- Handling missing values through imputation and row removal strategies.
- Encoding categorical features.
- Exploratory data analysis (EDA) to understand distributions and correlations.
- Addressing severe class imbalance using *ADASYN* resampling.
- Model training using *Random Forest* and *XGBoost* classifiers.
- Hyperparameter optimization via *GridSearchCV* with cross-validation.
- Model evaluation based on *ROC-AUC, **sensitivity, **specificity*, and other metrics.
- Feature importance and permutation importance analysis for interpretability.

---

## Requirements

All experiments were performed in a Python environment.  
The main dependencies (as listed in the thesis *Appendix A*) are:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imblearn
- joblib

You can install the required libraries using:


```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imblearn joblib
```

---

## How to Run

1. Clone or download the repository.
2. Ensure you have installed all required Python libraries.
3. First, open and run the **cerebral_stroke_prediction.ipynb** notebook:
   - This notebook handles data loading, cleaning, preprocessing, resampling, model training, and saves the cleaned datasets.
4. Then, open and run the **EDA_of_cerebral_stroke_prediction.ipynb** notebook:
   - This notebook loads the saved cleaned dataset and performs exploratory data analysis (EDA) and visualizations.

---

## Dataset

The dataset used in this study is the *Cerebral Stroke Prediction - Imbalanced Dataset* sourced from Kaggle:  
[https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)

- Number of records: 43,400 patients
- Number of features: 10 (plus target label)
- Stroke prevalence: 2.1%

---

## Acknowledgements

- Dataset originally compiled by Liu, Fan, and Wu (2019).
- Sampling technique inspired by Chawla et al. (2002) and subsequent literature on ADASYN.
- Modeling strategies referenced from multiple contemporary studies (2019–2024) reviewed in the thesis.

---
