# Telco Customer Churn Prediction
### A Learning Project | Machine Learning Classification with Python

> **Note:** This is an independent learning project built to develop hands-on skills in machine learning, feature engineering, and predictive analytics using a real-world telecom dataset from Kaggle.

---

## What I Learned & Built

Customer churn — when subscribers cancel their service — is one of the most studied problems in business analytics. I chose this dataset because it mirrors real challenges in my domain (health services, subscription-based operations) and helped me understand how to build end-to-end ML workflows: from raw data to a tuned, production-ready classification model.

Working through this helped me understand:
- How to identify and fix data quality issues before modelling
- How to engineer meaningful features that improve model performance
- How to evaluate models beyond accuracy (precision, recall, F1)
- Why hyperparameter tuning matters and how to do it systematically

---

## Project Overview

A classification model to predict which customers are likely to churn, built using Python and scikit-learn. Two algorithms were tested and tuned — **Decision Tree** and **K-Nearest Neighbours (KNN)** — with the Decision Tree achieving **79.2% accuracy** after hyperparameter tuning.

**Dataset:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
- 7,043 customer records × 21 features
- Target: `Churn` (Yes / No) — 73.4% No, 26.6% Yes (imbalanced)

---

## Project Workflow

```
Raw Data
   │
   ├── Exploratory Data Analysis
   ├── Data Quality Fixes
   │     ├── Fixed TotalCharges dtype (object → float)
   │     └── Dropped 11 rows with missing values
   ├── Feature Engineering
   │     ├── TenureGroup (binned into 1–6 year bands)
   │     ├── AvgChargesPerMonth (TotalCharges / tenure)
   │     ├── ServiceUsageScore (count of active services)
   │     └── DependentsTenureRatio
   ├── Feature Selection
   │     ├── Correlation matrix → dropped redundant features
   │     └── Lasso regularization (alpha=0.01) → 9 key features identified
   ├── Encoding & Scaling
   │     ├── Binary encoding for Yes/No columns
   │     ├── One-hot encoding for multi-class categoricals
   │     └── MinMaxScaler on numerical features
   ├── Train/Test Split (80/20)
   └── Model Training & Hyperparameter Tuning
         ├── Decision Tree (GridSearch: depth, criterion)
         └── KNN (GridSearch: k, metric, weights)
```

---

## Key Findings from EDA

| Factor | Churn Pattern |
|---|---|
| Short tenure (new customers) | Higher churn risk |
| Month-to-month contracts | Higher churn risk |
| Electronic check payment | Higher churn risk |
| No online security / tech support | Higher churn risk |
| Senior citizens | Higher churn risk |
| Long-term contracts (1–2 year) | Lower churn risk |
| Higher monthly charges | Higher churn risk |

---

## Model Results

### Before Tuning

| Metric | Decision Tree | KNN |
|---|---|---|
| Accuracy | 73.1% | 75.9% |
| Precision | 0.48 | 0.54 |
| Recall | 0.51 | 0.51 |
| F1 Score | 0.50 | 0.53 |

### After Hyperparameter Tuning

| Metric | Decision Tree | KNN |
|---|---|---|
| Accuracy | **79.2%** | 77.6% |
| Precision | **0.638** | 0.627 |
| Recall | **0.576** | 0.413 |
| F1 Score | **0.605** | 0.498 |

**Best Settings:**
- Decision Tree: `max_depth=6`, `criterion='entropy'`
- KNN: `n_neighbors=10`, `metric='minkowski'`, `weights='uniform'`

**Winner: Decision Tree** — outperformed KNN across all metrics after tuning.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Modelling, feature selection, scaling, tuning |
| Matplotlib, Seaborn | EDA visualisation |
| Google Colab | Development environment |

---

## Key Takeaways

**What this project taught me that applies to real analytics work:**
- Feature engineering often matters more than algorithm choice — the custom features (ServiceUsageScore, TenureGroup) improved model interpretability significantly
- Accuracy alone is misleading on imbalanced datasets — F1 and Recall are more meaningful when the minority class (churners) is what you care about
- Lasso regularization is a practical, fast method for feature selection before modelling
- Hyperparameter tuning improved accuracy by ~6 percentage points — not huge, but meaningful in a business context

---

## Future Improvements
- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Try ensemble models (Random Forest, XGBoost, LightGBM)
- [ ] Use cross-validation instead of single train/test split
- [ ] Add SHAP analysis for feature interpretability
- [ ] Build a Streamlit dashboard for live predictions

---

## How to Run

```bash
git clone https://github.com/KP-Kranthi/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
pip install pandas numpy scikit-learn matplotlib seaborn
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) and place as `WA_Fn-UseC_-Telco-Customer-Churn.csv`, then open the notebook in Jupyter or [Google Colab](https://colab.research.google.com/drive/1pDARUXZXsfe9-TLGirZHK4uIfohs0M9n?usp=sharing).

---

*Part of my ongoing self-directed learning in data analytics and machine learning. See my other projects: [E-Commerce Data Pipeline on Databricks](https://github.com/KP-Kranthi/Ecommerce-pipeline-databricks) 
