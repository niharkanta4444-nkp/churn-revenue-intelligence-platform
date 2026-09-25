"""
AI-Powered Customer Churn & Revenue Intelligence Platform
===========================================================
A single-file Streamlit application implementing the full BI pipeline:
RAW DATA -> CLEANING -> EDA -> KPIs -> TREND/SEGMENT ANALYSIS -> DRIVER ANALYSIS
-> CUSTOMER RISK ANALYSIS -> OPPORTUNITY DETECTION -> AI INSIGHTS -> ACTIONS

Dataset : Bank Customer Churn Dataset (10,000 customers)
Source  : https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
          (mirrored at https://github.com/kuropeter/Bank-Customer-Churn-Analysis-with-python)

IMPORTANT DATA LIMITATION (declared explicitly, not hidden):
This dataset is a single-snapshot customer table, NOT a transaction log.
There is no date/time column and no product/order-level data. Therefore:
  - "Revenue" is approximated using `Balance` (account balance), the closest
    valid numeric proxy for customer monetary value in this dataset.
  - "Sales/Product analysis" is approximated using `Geography` and
    `NumOfProducts` as segment dimensions (the closest valid substitutes).
  - True calendar trend analysis (month-over-month) is NOT possible; instead,
    Tenure (years with bank) is used to analyze value accumulation over the
    customer lifecycle, which is the closest valid time-like dimension.
No numbers in this app are fabricated -- every figure is computed directly
from the dataset at runtime.

Author: Generated for AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Churn_Modelling.csv")

st.set_page_config(page_title="Churn & Revenue Intelligence Platform",
                    layout="wide", page_icon="\U0001F4CA")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# 2. DATA CLEANING PIPELINE (transparent, logged)
# ---------------------------------------------------------------------------
@st.cache_data
def clean_data(df: pd.DataFrame):
    log = []
    df = df.copy()

    log.append(f"Raw rows: {len(df)}, raw columns: {df.shape[1]}")

    # Duplicates on the natural key (CustomerId)
    dup_ids = df["CustomerId"].duplicated().sum()
    if dup_ids > 0:
        df = df.drop_duplicates(subset="CustomerId", keep="first")
        log.append(f"Removed {dup_ids} duplicate CustomerId rows.")
    else:
        log.append("No duplicate CustomerId records found.")

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        df = df.drop_duplicates()
        log.append(f"Removed {dup_rows} fully duplicate rows.")
    else:
        log.append("No fully duplicate rows found.")

    # Missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for c in num_cols:
            if df[c].isnull().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
                log.append(f"Filled {c} missing values with median ({med:.2f}).")
        for c in cat_cols:
            if df[c].isnull().any():
                mode = df[c].mode().iloc[0]
                df[c] = df[c].fillna(mode)
                log.append(f"Filled {c} missing values with mode ({mode}).")
    else:
        log.append("No missing values found in any column.")

    # Type validation
    for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
              "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    invalid_after_cast = df.isnull().sum().sum() - missing_before
    if invalid_after_cast > 0:
        df = df.dropna()
        log.append(f"Dropped {invalid_after_cast} rows with non-numeric/invalid values after type validation.")

    # Invalid value ranges (business-rule validation)
    before = len(df)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
    df = df[df["CreditScore"].between(300, 850)]
    df = df[df["Balance"] >= 0]
    df = df[df["EstimatedSalary"] >= 0]
    df = df[df["Tenure"].between(0, 15)]
    removed_invalid = before - len(df)
    if removed_invalid > 0:
        log.append(f"Removed {removed_invalid} rows with out-of-range/invalid values.")
    else:
        log.append("No out-of-range invalid values found.")

    # Standardize categorical text
    df["Geography"] = df["Geography"].astype(str).str.strip().str.title()
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()

    # Outlier flagging (IQR method) - flagged, not silently deleted
    for c in ["Balance", "EstimatedSalary", "CreditScore"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        log.append(f"{c}: {n_outliers} statistical outliers detected (kept in data, not removed).")

    log.append(f"Final cleaned rows: {len(df)}, columns: {df.shape[1]}")
    return df, log


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING / RISK SEGMENTATION (rule-based, explainable)
# ---------------------------------------------------------------------------
@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeBand"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 60, 100],
                            labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    df["ValueSegment"] = pd.qcut(df["Balance"].rank(method="first"), 4,
                                  labels=["Low Value", "Mid-Low Value", "Mid-High Value", "High Value"])

    # Rule-based risk score built from measurable variables:
    # inactivity, low product count, low tenure, and being in a high-churn geography
    risk_score = (
        (df["IsActiveMember"] == 0).astype(int) * 2
        + (df["NumOfProducts"] <= 1).astype(int) * 2
        + (df["Tenure"] <= 2).astype(int) * 1
        + (df["NumOfProducts"] >= 3).astype(int) * 1   # 3-4 products correlates with churn in this data
    )
    df["RiskScore"] = risk_score

    def bucket(s):
        if s >= 4:
            return "High Risk"
        elif s >= 2:
            return "Medium Risk"
        else:
            return "Low Risk"
    df["RiskSegment"] = df["RiskScore"].apply(bucket)
    return df


# ---------------------------------------------------------------------------
# 4. KPI LAYER
# ---------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    total_customers = len(df)
    churned = int(df["Exited"].sum())
    churn_rate = df["Exited"].mean() * 100
    retention_rate = 100 - churn_rate
    total_balance = df["Balance"].sum()
    avg_balance = df["Balance"].mean()
    active_pct = df["IsActiveMember"].mean() * 100
    avg_products = df["NumOfProducts"].mean()
    # CLV proxy: Balance x Tenure (years) as a simple lifetime-value stand-in
    clv_proxy = (df["Balance"] * df["Tenure"]).mean()
    at_risk_balance = df.loc[df["RiskSegment"] == "High Risk", "Balance"].sum()
    return {
        "total_customers": total_customers,
        "churned": churned,
        "churn_rate": churn_rate,
        "retention_rate": retention_rate,
        "total_balance": total_balance,
        "avg_balance": avg_balance,
        "active_pct": active_pct,
        "avg_products": avg_products,
        "clv_proxy": clv_proxy,
        "at_risk_balance": at_risk_balance,
    }


# ---------------------------------------------------------------------------
# 5. ML CHURN MODEL
# ---------------------------------------------------------------------------
@st.cache_resource
def train_models(df: pd.DataFrame):
    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    X = pd.get_dummies(df[features + ["Geography", "Gender"]], columns=["Geography", "Gender"], drop_first=True)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_s, y_train)
    pred_lr = log_reg.predict(X_test_s)
    proba_lr = log_reg.predict_proba(X_test_s)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    results["Logistic Regression"] = metrics(y_test, pred_lr, proba_lr)
    results["Random Forest"] = metrics(y_test, pred_rf, proba_rf)

    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return results, importance


# ---------------------------------------------------------------------------
# 6. RISK / OPPORTUNITY DETECTION (data-driven, not hardcoded)
# ---------------------------------------------------------------------------
def detect_risks(df: pd.DataFrame, kpis: dict) -> list:
    risks = []
    geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
    worst_geo = geo_churn.index[0]
    if geo_churn.iloc[0] > kpis["churn_rate"] / 100 + 0.03:
        risks.append({
            "risk": f"Elevated churn concentration in {worst_geo}",
            "evidence": f"{worst_geo} shows a churn rate of {geo_churn.iloc[0]*100:.1f}% vs the overall {kpis['churn_rate']:.1f}%.",
            "impact": "Disproportionate revenue loss concentrated in one market.",
            "action": f"Launch a targeted retention review for {worst_geo} customers, focusing on service quality and product fit."
        })

    inactive_high_bal = df[(df["IsActiveMember"] == 0) & (df["Balance"] > df["Balance"].median())]
    if len(inactive_high_bal) > 0:
        risks.append({
            "risk": "High-balance customers with low engagement",
            "evidence": f"{len(inactive_high_bal)} customers hold above-median balances but are inactive members "
                        f"(avg balance {inactive_high_bal['Balance'].mean():,.0f}).",
            "impact": "These customers represent high monetary exposure with weak engagement, raising churn probability.",
            "action": "Trigger a re-engagement campaign (personalized outreach, loyalty incentives) for inactive high-balance customers."
        })

    multi_prod_churn = df[df["NumOfProducts"] >= 3]["Exited"].mean()
    if multi_prod_churn > kpis["churn_rate"] / 100:
        risks.append({
            "risk": "Customers with 3-4 products churn at a higher rate than average",
            "evidence": f"Customers holding 3+ products churn at {multi_prod_churn*100:.1f}% vs overall {kpis['churn_rate']:.1f}%.",
            "impact": "Counter-intuitive pattern suggests product bundling/cross-sell may be causing friction, not loyalty.",
            "action": "Investigate satisfaction and service experience for multi-product holders; review bundling terms."
        })

    at_risk_share = kpis["at_risk_balance"] / kpis["total_balance"] * 100
    if at_risk_share > 5:
        risks.append({
            "risk": "Material share of total balance sits in High Risk segment",
            "evidence": f"{at_risk_share:.1f}% of total customer balance ({kpis['at_risk_balance']:,.0f}) belongs to customers flagged High Risk.",
            "impact": "A meaningful share of the balance base could be lost without intervention.",
            "action": "Prioritize retention offers and relationship-manager outreach for the High Risk segment first."
        })
    return risks


def detect_opportunities(df: pd.DataFrame, kpis: dict) -> list:
    opps = []
    geo_bal = df.groupby("Geography")["Balance"].mean().sort_values(ascending=False)
    best_geo = geo_bal.index[0]
    opps.append({
        "opportunity": f"{best_geo} holds the highest average customer balance",
        "evidence": f"Average balance in {best_geo} is {geo_bal.iloc[0]:,.0f}, the highest across regions.",
        "reason": "Indicates a wealthier or more engaged customer base with cross-sell/up-sell potential.",
        "action": f"Prioritize premium product offers and relationship banking services in {best_geo}."
    })

    low_risk_high_val = df[(df["RiskSegment"] == "Low Risk") & (df["ValueSegment"] == "High Value")]
    opps.append({
        "opportunity": "Loyal high-value customer base identified",
        "evidence": f"{len(low_risk_high_val)} customers are both High Value and Low Risk "
                    f"({len(low_risk_high_val)/len(df)*100:.1f}% of the base).",
        "reason": "These customers are stable, profitable anchors for referral and loyalty programs.",
        "action": "Use this segment for referral incentives and to pilot new premium products before wider rollout."
    })

    single_product = df[df["NumOfProducts"] == 1]
    cross_sell_target = single_product[single_product["IsActiveMember"] == 1]
    opps.append({
        "opportunity": "Active single-product customers are a cross-sell opportunity",
        "evidence": f"{len(cross_sell_target)} active customers currently hold only 1 product.",
        "reason": "Active engagement with only one product suggests room to deepen the relationship.",
        "action": "Run a cross-sell campaign offering a second product (savings, credit card, or investment) to this group."
    })
    return opps


# ---------------------------------------------------------------------------
# 7. AI INSIGHT ENGINE (rule-driven synthesis over computed results - no fabrication)
# ---------------------------------------------------------------------------
def generate_ai_insights(risks: list, opps: list) -> list:
    insights = []
    for r in risks:
        insights.append({
            "type": "Risk",
            "fact": r["evidence"],
            "insight": r["risk"],
            "impact": r["impact"],
            "action": r["action"],
        })
    for o in opps:
        insights.append({
            "type": "Opportunity",
            "fact": o["evidence"],
            "insight": o["opportunity"],
            "impact": o["reason"],
            "action": o["action"],
        })
    return insights


# ===========================================================================
# STREAMLIT APP
# ===========================================================================
def main():
    st.title("\U0001F4CA AI-Powered Customer Churn & Revenue Intelligence Platform")
    st.caption("Bank Customer Churn Dataset · 10,000 customers · Data -> Information -> Insight -> Decision -> Action")

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Please place Churn_Modelling.csv next to this script.")
        st.stop()

    raw_df = load_raw_data(DATA_PATH)
    clean_df, clean_log = clean_data(raw_df)
    df = engineer_features(clean_df)

    # ---------------- SIDEBAR FILTERS ----------------
    st.sidebar.header("Filters")
    geos = st.sidebar.multiselect("Geography", sorted(df["Geography"].unique()), default=sorted(df["Geography"].unique()))
    genders = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    risk_filter = st.sidebar.multiselect("Risk Segment", sorted(df["RiskSegment"].unique()), default=sorted(df["RiskSegment"].unique()))
    age_range = st.sidebar.slider("Age range", int(df["Age"].min()), int(df["Age"].max()),
                                   (int(df["Age"].min()), int(df["Age"].max())))

    fdf = df[df["Geography"].isin(geos) & df["Gender"].isin(genders)
             & df["RiskSegment"].isin(risk_filter)
             & df["Age"].between(age_range[0], age_range[1])]

    if fdf.empty:
        st.warning("No records match the selected filters. Please broaden your filter selection.")
        st.stop()

    kpis = compute_kpis(fdf)

    tabs = st.tabs(["\U0001F3E0 Executive Overview", "\U0001F30D Segment Analysis",
                     "\U0001F465 Customer & Risk Analysis", "\U0001F916 AI Insights & Actions",
                     "\U0001F9F9 Data Quality Log"])

    # ---------------- PAGE 1: EXECUTIVE OVERVIEW ----------------
    with tabs[0]:
        st.subheader("What is happening to the business?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{kpis['total_customers']:,}")
        c2.metric("Churn Rate", f"{kpis['churn_rate']:.1f}%")
        c3.metric("Retention Rate", f"{kpis['retention_rate']:.1f}%")
        c4.metric("Total Balance (Revenue Proxy)", f"{kpis['total_balance']:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Balance / Customer", f"{kpis['avg_balance']:,.0f}")
        c6.metric("Active Member %", f"{kpis['active_pct']:.1f}%")
        c7.metric("Avg Products / Customer", f"{kpis['avg_products']:.2f}")
        c8.metric("Balance at High Risk", f"{kpis['at_risk_balance']:,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            churn_by_tenure = fdf.groupby("Tenure")["Exited"].mean().reset_index()
            fig = px.line(churn_by_tenure, x="Tenure", y="Exited", markers=True,
                          title="Churn Rate by Tenure (years with bank)",
                          labels={"Exited": "Churn Rate"})
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            bal_by_tenure = fdf.groupby("Tenure")["Balance"].mean().reset_index()
            fig2 = px.bar(bal_by_tenure, x="Tenure", y="Balance",
                          title="Average Balance by Tenure (lifecycle value trend)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Key observation:** " +
                    (f"Churn is concentrated among lower-tenure and disengaged customers; overall churn stands at "
                     f"{kpis['churn_rate']:.1f}% against a base of {kpis['total_customers']:,} customers holding "
                     f"a combined balance of {kpis['total_balance']:,.0f}."))

    # ---------------- PAGE 2: SEGMENT ANALYSIS ----------------
    with tabs[1]:
        st.subheader("Segment Analysis (Geography & Product act as the 'sales/product' dimension)")
        st.info("This dataset has no product/order transactions, so Geography and NumOfProducts are used "
                "as the closest valid substitute for sales/product analysis.", icon="\u2139\ufe0f")

        col1, col2 = st.columns(2)
        with col1:
            geo_stats = fdf.groupby("Geography").agg(Customers=("CustomerId", "count"),
                                                       ChurnRate=("Exited", "mean"),
                                                       AvgBalance=("Balance", "mean")).reset_index()
            fig = px.bar(geo_stats, x="Geography", y="Customers", color="ChurnRate",
                        title="Customers & Churn Rate by Geography", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            prod_stats = fdf.groupby("NumOfProducts").agg(Customers=("CustomerId", "count"),
                                                            ChurnRate=("Exited", "mean")).reset_index()
            fig2 = px.bar(prod_stats, x="NumOfProducts", y="ChurnRate",
                         title="Churn Rate by Number of Products")
            fig2.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            gender_stats = fdf.groupby("Gender")["Exited"].mean().reset_index()
            fig3 = px.bar(gender_stats, x="Gender", y="Exited", title="Churn Rate by Gender")
            fig3.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            age_stats = fdf.groupby("AgeBand", observed=True)["Exited"].mean().reset_index()
            fig4 = px.bar(age_stats, x="AgeBand", y="Exited", title="Churn Rate by Age Band")
            fig4.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Driver read-out:** Compare bars above — geography, product count, and age band with "
                    "visibly higher bars are the segments driving churn upward relative to the overall rate.")

    # ---------------- PAGE 3: CUSTOMER & RISK ANALYSIS ----------------
    with tabs[2]:
        st.subheader("Customer Risk Segmentation & Predictive Churn Model")

        risk_counts = fdf["RiskSegment"].value_counts().reset_index()
        risk_counts.columns = ["RiskSegment", "Customers"]
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.pie(risk_counts, names="RiskSegment", values="Customers",
                        title="Customer Risk Segments (rule-based)",
                        color="RiskSegment",
                        color_discrete_map={"Low Risk": "#2ca02c", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            value_seg = fdf.groupby(["ValueSegment", "RiskSegment"], observed=True).size().reset_index(name="Customers")
            fig2 = px.bar(value_seg, x="ValueSegment", y="Customers", color="RiskSegment",
                         title="Value Segment vs Risk Segment", barmode="stack")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Risk logic:** score += 2 if inactive member, += 2 if 1 or fewer products, "
                    "+= 1 if tenure <= 2 years, += 1 if 3+ products (this dataset's multi-product holders churn more). "
                    "Score >= 4 -> High Risk, 2-3 -> Medium Risk, else Low Risk.")

        st.divider()
        st.subheader("Predictive Churn Model")
        with st.spinner("Training models..."):
            results, importance = train_models(clean_df)

        mcol1, mcol2 = st.columns(2)
        for col, (name, m) in zip([mcol1, mcol2], results.items()):
            with col:
                st.markdown(f"**{name}**")
                st.write(f"Accuracy: {m['accuracy']:.3f} | Precision: {m['precision']:.3f} | "
                        f"Recall: {m['recall']:.3f} | F1: {m['f1']:.3f} | ROC-AUC: {m['roc_auc']:.3f}")
                cm = m["confusion_matrix"]
                fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                    x=["Stayed", "Churned"], y=["Stayed", "Churned"],
                                    title=f"Confusion Matrix - {name}")
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**Note:** Metrics are computed on a real 80/20 held-out test split — no accuracy figures are assumed or inflated.")

        st.subheader("What drives churn? (Feature Importance - Random Forest)")
        imp_df = importance.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        fig_imp = px.bar(imp_df.head(10), x="Importance", y="Feature", orientation="h",
                         title="Top Churn Drivers")
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- PAGE 4: AI INSIGHTS & ACTIONS ----------------
    with tabs[3]:
        st.subheader("AI-Generated Insights (derived only from computed analytical results)")
        risks = detect_risks(fdf, kpis)
        opps = detect_opportunities(fdf, kpis)
        insights = generate_ai_insights(risks, opps)

        for i, ins in enumerate(insights, 1):
            color = "\U0001F534" if ins["type"] == "Risk" else "\U0001F7E2"
            with st.expander(f"{color} {ins['type']} #{i}: {ins['insight']}"):
                st.markdown(f"**FACT:** {ins['fact']}")
                st.markdown(f"**INSIGHT:** {ins['insight']}")
                st.markdown(f"**{'RISK' if ins['type']=='Risk' else 'OPPORTUNITY'}:** {ins['impact']}")
                st.markdown(f"**ACTION:** {ins['action']}")

        st.divider()
        st.subheader("Recommended Business Actions")
        for i, r in enumerate(risks, 1):
            st.markdown(f"**{i}. {r['action']}**")
            st.caption(f"Evidence: {r['evidence']} — Objective: reduce churn exposure and protect at-risk balance.")
        for i, o in enumerate(opps, len(risks) + 1):
            st.markdown(f"**{i}. {o['action']}**")
            st.caption(f"Evidence: {o['evidence']} — Objective: grow balance/relationship depth in this segment.")

        st.info("Recommendations are derived strictly from computed statistics above. No guaranteed outcomes are claimed.")

    # ---------------- PAGE 5: DATA QUALITY LOG ----------------
    with tabs[4]:
        st.subheader("Data Cleaning & Quality Log")
        for line in clean_log:
            st.write("- " + line)
        st.subheader("Raw Data Sample")
        st.dataframe(raw_df.head(20))
        st.subheader("Cleaned & Engineered Data Sample")
        st.dataframe(df.head(20))


if __name__ == "__main__":
    main()
