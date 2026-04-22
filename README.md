#  Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company using Decision Tree and K-Nearest Neighbours (KNN) classifiers.

---

##  Table of Contents

- [Business Problem](#business-problem)
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Models & Results](#models--results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

##  Business Problem

Customer churn — when customers stop doing business with a company — is a costly problem, especially in subscription-based industries like telecom. Acquiring new customers is significantly more expensive than retaining existing ones.

This project uses predictive analytics to identify customers who are likely to churn, enabling the business to take proactive retention actions such as targeted offers, loyalty programs, or improved customer service.

---

##  Project Goal

Build a classification model that predicts whether a customer will churn based on historical data including tenure, contract type, monthly charges, and service usage patterns.

---

##  Dataset

- **Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
- **Size:** 7,043 rows × 21 columns
- **Target Variable:** `Churn` (Yes / No)
- **Class Balance:** ~73.4% No Churn, ~26.6% Churn (imbalanced)

---

##  Project Workflow

```
Raw Data
   │
   ├── Data Exploration (dataset.info, dataset.head)
   ├── Data Quality Handling
   │     ├── Fixed TotalCharges dtype (object → float)
   │     └── Dropped 11 missing rows (TotalCharges)
   ├── EDA & Distribution Analysis
   ├── Data Processing
   │     ├── Binary encoding (gender, Yes/No columns)
   │     ├── Replaced "No internet/phone service" → "No"
   │     └── One-hot encoding (InternetService, Contract, PaymentMethod)
   ├── Feature Engineering
   │     ├── Dropped customerID
   │     ├── TenureGroup (binned into 1–6 year bands)
   │     ├── AvgChargesPerMonth (TotalCharges / tenure)
   │     ├── ServiceUsageScore (sum of active services)
   │     └── DependentsTenureRatio (Dependents / tenure)
   ├── Feature Selection
   │     ├── Correlation matrix → dropped gender, AvgChargesPerMonth
   │     └── Lasso regularization (alpha=0.01) → 9 key features identified
   ├── Scaling (MinMaxScaler on numerical features)
   ├── Train/Test Split (80/20)
   └── Model Building & Hyperparameter Tuning
         ├── Decision Tree (best: depth=6, criterion=entropy)
         └── KNN (best: k=10, metric=minkowski, weights=uniform)
```

---

##  Key Findings from EDA

| Factor | Churn Tendency |
|---|---|
| Short tenure (new customers) | ↑ Higher churn |
| Month-to-month contracts | ↑ Higher churn |
| Electronic check payment | ↑ Higher churn |
| Fibre optic internet service | ↑ Higher churn |
| No online security / tech support | ↑ Higher churn |
| Senior citizens | ↑ Higher churn |
| No partner or dependents | ↑ Higher churn |
| Higher monthly charges | ↑ Higher churn |
| Long-term contracts (1–2 year) | ↓ Lower churn |
| Low service usage score | ↓ Lower engagement, higher churn risk |

---

##  Models & Results

### Before Hyperparameter Tuning

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
- **Decision Tree:** `max_depth=6`, `criterion='entropy'`
- **KNN:** `n_neighbors=10`, `metric='minkowski'`, `weights='uniform'`, `p=2`

###  Winner: Decision Tree

The tuned Decision Tree outperforms KNN across all metrics — accuracy, precision, recall, and F1 score — making it the recommended model for this problem.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Environment:** Google Colab

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) and place it in the project root as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

4. Run the notebook:
   ```bash
   jupyter notebook Telco_Customer_Churn_Prediction.ipynb
   ```

   Or open directly in [Google Colab](https://colab.research.google.com/drive/1pDARUXZXsfe9-TLGirZHK4uIfohs0M9n?usp=sharing).

---

##  Future Improvements

- [ ] Handle class imbalance using SMOTE or class weighting
- [ ] Try ensemble models (Random Forest, XGBoost, LightGBM)
- [ ] Use cross-validation instead of a single train/test split
- [ ] Perform SHAP analysis for feature interpretability
- [ ] Build a simple Streamlit dashboard for predictions

---

##  Project Structure

```
telco-churn-prediction/
│
├── Telco_Customer_Churn_Prediction.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset (download from Kaggle)
└── README.md
```

---

*Project completed as part of MBA coursework.*
