# AI-Powered Customer Churn & Revenue Intelligence Platform

## 1. Project Title
AI-Powered Customer Churn & Revenue Intelligence Platform — Bank Customer Dataset

## 2. Problem Statement
Banks lose recurring revenue when customers churn, but raw transactional/account data alone does not tell management *why* customers leave, *which* customers are at risk, or *what* to do about it. This project converts raw customer data into KPIs, drivers, risk segments, and AI-generated, evidence-based business actions.

## 3. Objective
Build a decision-support platform answering: What is happening? How is it changing? Why is it happening? Who is driving it? What risks/opportunities exist? What should management do?

## 4. Business Use Case
A retail bank wants to reduce customer attrition and protect its balance/deposit base. This platform gives a business user (not a data scientist) a self-service view of churn drivers, at-risk customers, and concrete retention/growth actions.

## 5. Dataset Description
**Bank Customer Churn Dataset** — 10,000 customers, 14 columns, no missing values, no duplicates.

| Column | Description |
|---|---|
| CustomerId, Surname, RowNumber | Identifiers (not used as model features) |
| CreditScore | Customer's credit score (300–850) |
| Geography | Country: France, Germany, Spain |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years as a bank customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held (1–4) |
| HasCrCard | Has a credit card (0/1) |
| IsActiveMember | Active membership flag (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | Churn label (1 = churned, 0 = retained) |

## 6. Dataset Source
Kaggle: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
(Mirror used for programmatic download: https://github.com/kuropeter/Bank-Customer-Churn-Analysis-with-python/blob/main/Churn_Modelling.csv)

## 7. Important Data Limitation (declared explicitly)
This dataset is a **single-snapshot customer table**, not a transaction log — there is no date column and no product/order-level data. As a result:
- "Revenue" is approximated using **Balance** (closest valid monetary proxy).
- "Sales/Product analysis" is approximated using **Geography** and **NumOfProducts** as segment dimensions.
- True calendar-based trend analysis is not possible; **Tenure** (years with bank) is used instead as the closest valid lifecycle/time-like dimension.
No figures in this project are fabricated — every number is computed live from the data.

## 8. Technologies Used
Python 3.10+, Streamlit, Pandas, NumPy, Plotly, scikit-learn (Logistic Regression, Random Forest)

## 9. Project Architecture
```
Churn_Modelling.csv
      │
      ▼
Data Loading → Data Cleaning (dedupe, missing values, type/range validation, outlier flagging)
      │
      ▼
Feature Engineering (Age Band, Value Segment, rule-based Risk Score/Segment)
      │
      ▼
KPI Layer → Executive Overview (Page 1)
      │
      ▼
Segment/Driver Analysis (Page 2) → Customer Risk & ML Churn Model (Page 3)
      │
      ▼
Risk Detection + Opportunity Detection (data-driven)
      │
      ▼
AI Insight Engine (FACT → INSIGHT → RISK/OPPORTUNITY → ACTION) (Page 4)
      │
      ▼
Recommended Business Actions
```

## 10. Data-Cleaning Process
- Removed duplicate `CustomerId` and fully duplicate rows (none found in this dataset).
- Median/mode imputation for any missing numeric/categorical values (none found).
- Type validation: forced numeric columns to numeric type; dropped any row that failed validation.
- Business-rule range validation: Age 18–100, CreditScore 300–850, Balance ≥ 0, Salary ≥ 0, Tenure 0–15.
- Standardized text casing for `Geography` and `Gender`.
- Outliers detected via IQR method and **flagged/reported**, not silently deleted.
- Full cleaning log is displayed live in the app's "Data Quality Log" tab.

## 11. KPI Explanation
- **Churn Rate / Retention Rate** — % of customers who exited vs. stayed.
- **Total Balance / Avg Balance** — revenue-proxy KPIs.
- **Active Member %** — engagement KPI.
- **Avg Products per Customer** — cross-sell depth KPI.
- **CLV Proxy** — Balance × Tenure, a simple lifetime-value stand-in (no transaction history exists for a true CLV).
- **Balance at High Risk** — total balance held by customers flagged High Risk, i.e. revenue exposed to churn.

## 12. Analytical Methodology
- **KPI computation** on filtered data (live, driven by sidebar filters).
- **Segment analysis**: churn/balance by Geography, NumOfProducts, Gender, Age Band.
- **Driver analysis**: Random Forest feature importance ranks the strongest churn predictors.
- **Rule-based risk scoring**: transparent point system built only from measurable variables (inactivity, low product count, low tenure, 3+ products).
- **Risk/Opportunity detection**: threshold-based rules compare segment churn/balance against overall averages; only fires when supported by the data.

## 13. ML Methodology
Two interpretable models are trained on an 80/20 stratified train/test split:
- **Logistic Regression** (scaled features)
- **Random Forest** (300 trees, class-balanced)

Reported metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix — all computed on the held-out test set, no inflated claims.

## 14. AI Methodology
An AI Insight Engine converts the risk/opportunity detection results (never raw unprocessed rows) into structured FACT → INSIGHT → RISK/OPPORTUNITY → ACTION cards. The insight text is generated only from the analytical results already computed in the pipeline, so no fact is invented beyond what the data supports.

## 15. Dashboard Explanation
- **Page 1 – Executive Overview**: KPI cards + churn/balance trend by tenure.
- **Page 2 – Segment Analysis**: churn/balance broken down by geography, product count, gender, age band.
- **Page 3 – Customer & Risk Analysis**: risk segment pie/stacked charts, ML model comparison, feature importance.
- **Page 4 – AI Insights & Actions**: FACT/INSIGHT/RISK/ACTION cards and a consolidated action list.
- **Page 5 – Data Quality Log**: full transparency on cleaning steps and raw vs. cleaned data samples.

Filters available: Geography, Gender, Risk Segment, Age range (all exist natively in the dataset).

## 16. How to Install
```bash
git clone <this-repo>
cd <this-repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 17. How to Run
Ensure `Churn_Modelling.csv` is in the same folder as `churn_intelligence_platform.py`, then:
```bash
streamlit run churn_intelligence_platform.py
```
Open the local URL Streamlit prints (typically http://localhost:8501).

## 18. Example Usage
1. Launch the app.
2. Use sidebar filters to focus on, e.g., Germany + High Risk customers.
3. Read KPIs and trend charts on the Executive Overview tab.
4. Check the AI Insights tab for evidence-backed risks, opportunities, and recommended actions.

## 19. Limitations
- No transactional/date data → no true calendar trend analysis or product-level sales figures.
- Risk segmentation is rule-based (explainable) rather than a separately tuned ML risk model.
- Dataset is a single historical snapshot; recommendations should be validated against current data before acting.

## 20. Future Improvements
- Integrate a transactional dataset to enable true time-series trend and product-level analysis.
- Add SHAP-based explainability for individual customer churn predictions.
- Connect the AI Insight Engine to a live LLM API for natural-language narrative generation on top of the structured facts.
- Add cohort-based retention curves once timestamped events are available.
