#  Telco Customer Churn Analysis

A full end-to-end data science project analyzing customer churn behavior for a telecommunications company. The project covers data cleaning, exploratory data analysis, and machine learning modeling — built across 3 structured assignments.

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

---

## Repository Structure

```
telco-customer-churn-analysis/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (downloaded from Kaggle)
├── churn_analysis.ipynb                    # Full Jupyter Notebook (all 3 assignments)
└── README.md
```

---

## Dataset Overview

The dataset contains **7,043 rows** and **21 columns** describing telecom customers and whether they churned.

| Feature | Description |
|---|---|
| `customerID` | Unique customer identifier (dropped before modeling) |
| `gender` | Male / Female |
| `SeniorCitizen` | Whether the customer is a senior (1 = Yes) |
| `Partner` / `Dependents` | Family status |
| `tenure` | Number of months with the company |
| `PhoneService`, `InternetService` | Service subscriptions |
| `OnlineSecurity`, `TechSupport` | Add-on services |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod` | How the customer pays |
| `MonthlyCharges`, `TotalCharges` | Billing information |
| `Churn` | **Target variable** — Yes / No |

---

## Project Walkthrough

### Assignment 1 — Data Exploration & Preparation

**Goal:** Understand the raw data and prepare it for machine learning.

**What was done:**

- Loaded the dataset and audited all column types using `.info()`, `.describe()`, and `.dtypes`
- Fixed `TotalCharges`: the column was stored as a string object — converted to numeric using `pd.to_numeric(..., errors='coerce')`. The 11 blank entries (new customers with zero tenure) were filled with `0`
- Dropped `customerID` — a unique identifier with no predictive value
- Binary-mapped all Yes/No columns to `1/0`, including treating `No internet service` and `No phone service` as `0` to keep features consistent
- Applied **One-Hot Encoding** (`pd.get_dummies`) to the three multi-class columns: `Contract`, `InternetService`, and `PaymentMethod` — used `drop_first=True` to avoid the dummy variable trap
- Removed 22 duplicate rows discovered after dropping the unique ID column
- Converted all columns to `float` for full ML compatibility
- Ran a **Data Quality Validation Report** confirming: 0 missing values, 0 duplicates, all numeric types

**Key finding:** The target variable `Churn` is imbalanced — only ~26% of customers churned. This was flagged early as a critical concern for modeling.

---

### Assignment 2 — Exploratory Data Analysis (EDA)

**Goal:** Discover patterns and risk factors linked to churn through visualization and statistics.

**What was done:**

**Demographic Analysis**
- Built a 2×2 grid of countplots comparing churn rates across `Gender`, `SeniorCitizen`, `Partner`, and `Dependents`
- Key finding: **Senior Citizens churn at 41.6%** vs 23.5% for non-seniors. Gender has almost no effect (~26% for both)

**Service & Contract Analysis**
- Reconstructed human-readable labels from dummy columns (e.g., `Contract_Two year` → `"Two Year"`) to produce clean bar charts
- Key finding: **Month-to-month contracts have ~42% churn** vs ~11% for one-year and ~3% for two-year contracts
- **Fiber Optic internet users churn at a significantly higher rate** than DSL users despite being a premium service
- Customers **without Online Security or Tech Support churn at nearly double the rate** of those with these services — these act as retention "anchors"

**Numeric Trends**
- Used side-by-side boxplots for `tenure`, `MonthlyCharges`, and `TotalCharges` split by churn status
- Key finding: Churned customers have a **median tenure of only 10 months** vs 38 months for retained customers. Churners also pay higher monthly fees (median $79.70 vs $64.50)

**Correlation Analysis**
- Computed Pearson correlations of all features against `Churn` and ranked by absolute value
- Top 5 churn predictors identified:
  1. `tenure` (strongest negative — longer customers stay, less likely to churn)
  2. `InternetService_Fiber optic` (positive — fiber users leave more)
  3. `Contract_Two year` (negative — long contracts retain customers)
  4. `PaymentMethod_Electronic check` (positive — electronic check users churn more)
  5. `TotalCharges` (negative — higher lifetime spend = more loyal)

**3 Key Business Insights:**

> **The New Customer Danger Zone** — Customers are most likely to leave within their first 10 months. Loyalty programs and onboarding support in year one are critical.

> **The Fiber Optic Paradox** — Despite being a premium product, Fiber has the highest churn rate. This signals potential pricing or service quality issues worth investigating.

> **Security as a Retention Tool** — Online Security and Tech Support act as anchors. Bundling these into month-to-month plans could significantly reduce that segment's churn.

---

### Assignment 3 — Churn Prediction Modeling

**Goal:** Build and compare machine learning models to predict which customers will churn.

**Methodology:**

- **Train/Test Split:** 80/20 with `stratify=y` to preserve the 26% churn ratio in both sets
- **Imbalance Handling:** `class_weight='balanced'` applied to both models so the algorithm pays more attention to the minority churn class
- **Feature Scaling:** `StandardScaler` applied to Logistic Regression only (tree-based models are scale-invariant)

**Models Trained:**

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline; interpretable; scaled features; balanced class weight |
| Random Forest | 100 trees; ensemble method; handles non-linearity; balanced class weight |

**Results:**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.7416 | 0.7794 |
| Recall (Churners) | **0.7796** | 0.4328 |
| Precision | 0.5079 | 0.6192 |
| F1 Score | **0.6151** | 0.5095 |
| AUC-ROC | **0.8397** | 0.8158 |

**Recommended Model: Logistic Regression**

Logistic Regression is the better model for this use case because it achieves a **Recall of 0.78** — meaning it correctly identifies 78% of customers who will churn, compared to only 43% for Random Forest. In churn prediction, a missed churner (false negative) is more costly than a false alarm, making Recall the most important metric. LR also leads on F1 Score and AUC-ROC.

> The addition of `class_weight='balanced'` boosted LR Recall from 0.52 → 0.78, at the cost of some accuracy (0.80 → 0.74). This is the right tradeoff for a retention-focused use case.

**Feature Importance (Random Forest):**

The top 3 churn drivers identified by the Random Forest:
1. **MonthlyCharges / TotalCharges** — Price sensitivity is the strongest signal. High-bill customers are most at risk
2. **Tenure** — New customers are the most vulnerable. Early engagement is critical
3. **Contract Type** — Month-to-month customers churn at far higher rates than those on long-term contracts

**Visualizations produced:**
- Confusion matrix heatmaps for both models
- ROC curve comparison with AUC scores
- Horizontal bar chart of top 10 feature importances

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/telco-customer-churn-analysis.git
cd telco-customer-churn-analysis
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Launch the notebook
```bash
jupyter notebook churn_analysis.ipynb
```

4. Run all cells in order — the notebook is self-contained and flows from Assignment 1 through Assignment 3

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Machine learning models and evaluation metrics |

---

## Key Results Summary

- **Best model:** Logistic Regression (with `class_weight='balanced'`)
- **Recall on churners:** 77.96%
- **AUC-ROC:** 0.8397
- **Top churn risk factors:** High monthly charges, low tenure, month-to-month contract
- **Highest-risk customer profile:** New customer, month-to-month contract, Fiber Optic internet, no Online Security, paying by electronic check

---

## Limitations & Next Steps

**Current limitations:**
- Model trained on historical data only — may not reflect future market behavior
- No external factors included (competitor pricing, market trends)
- Only structured tabular data used — no behavioral or text signals

**Planned improvements:**
- Apply SMOTE oversampling to improve minority class detection
- Hyperparameter tuning with GridSearchCV
- Threshold tuning to push Recall above 0.85
- Add time-based features (e.g., charge increase over tenure)
