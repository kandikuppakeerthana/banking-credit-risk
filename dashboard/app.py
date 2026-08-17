import streamlit as st
import pandas as pd
import requests
# Load cross-sell recommendation data
RECOMMENDATION_PATH = "notebooks/customer_cross_sell_recommendations.csv"

recommendation_df = pd.read_csv(RECOMMENDATION_PATH)
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Credit Risk System",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD FINAL BANKING DATA
# ============================================================

DATA_PATH = "notebooks/Phase_10_Final_Integrated_Banking_Output.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🏦 Banking Credit Risk & Cross-Sell System")

st.write(
    "An intelligent banking analytics platform for credit risk "
    "prediction, customer explainability, and product recommendations."
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🔮 Credit Risk Prediction",
        "🔍 Explainability",
        "🛍️ Product Recommendation"
    ]
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    st.write(
        "Overview of the Banking Credit Risk & Cross-Sell System"
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_customers = len(df)

    high_risk_customers = (
        df["Risk_Category"] == "High Risk"
    ).sum()

    low_risk_customers = (
        df["Risk_Category"] == "Low Risk"
    ).sum()

    avg_default_probability = (
        df["Default_Probability"].mean()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "🔴 High-Risk Customers",
            f"{high_risk_customers:,}"
        )

    with col3:
        st.metric(
            "🟢 Low-Risk Customers",
            f"{low_risk_customers:,}"
        )

    with col4:
        st.metric(
            "📊 Avg Default Probability",
            f"{avg_default_probability:.2%}"
        )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Risk Distribution")

    risk_counts = df["Risk_Category"].value_counts()

    st.bar_chart(risk_counts)

    # --------------------------------------------------------
    # DEFAULT PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📈 Default Probability Distribution")

    st.line_chart(
        df["Default_Probability"]
        .sort_values()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # BANKING DECISION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🏦 Banking Decision Distribution")

    decision_counts = (
        df["Banking_Decision"]
        .value_counts()
    )

    st.bar_chart(decision_counts)

    # --------------------------------------------------------
    # RECOMMENDED PRODUCT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🛍️ Recommended Product Distribution")

    product_counts = (
        df["Recommended_Products"]
        .str.split(", ")
        .explode()
        .value_counts()
    )

    product_counts = product_counts[
        product_counts.index != "No Recommendation"
    ]

    st.bar_chart(product_counts)


# ============================================================
# PAGE 2 — CREDIT RISK PREDICTION
# ============================================================

elif page == "🔮 Credit Risk Prediction":

    st.header("🔮 Credit Risk Prediction")

    st.write(
        "Enter customer financial and payment information "
        "to assess credit default risk."
    )

    # --------------------------------------------------------
    # CUSTOMER INPUT FORM
    # --------------------------------------------------------

    with st.form("credit_risk_form"):

        # ----------------------------------------------------
        # CUSTOMER INFORMATION
        # ----------------------------------------------------

        st.subheader("👤 Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            LIMIT_BAL = st.number_input(
                "Credit Limit (LIMIT_BAL)",
                min_value=0.0,
                value=20000.0,
                key="prediction_limit_bal"
            )

        with col2:

            SEX = st.number_input(
                "Gender (SEX)",
                min_value=1,
                max_value=2,
                value=2,
                step=1,
                key="prediction_sex"
            )

        with col3:

            EDUCATION = st.number_input(
                "Education (EDUCATION)",
                min_value=0,
                value=2,
                step=1,
                key="prediction_education"
            )

        col1, col2 = st.columns(2)

        with col1:

            MARRIAGE = st.number_input(
                "Marriage (MARRIAGE)",
                min_value=0,
                value=1,
                step=1,
                key="prediction_marriage"
            )

        with col2:

            AGE = st.number_input(
                "Age (AGE)",
                min_value=18,
                value=30,
                step=1,
                key="prediction_age"
            )

        # ----------------------------------------------------
        # PAYMENT HISTORY
        # ----------------------------------------------------

        st.subheader("💳 Payment History")

        col1, col2, col3 = st.columns(3)

        with col1:

            PAY_SEPT = st.number_input(
                "PAY_SEPT",
                value=0,
                step=1,
                key="prediction_pay_sept"
            )

            PAY_AUG = st.number_input(
                "PAY_AUG",
                value=0,
                step=1,
                key="prediction_pay_aug"
            )

        with col2:

            PAY_JUL = st.number_input(
                "PAY_JUL",
                value=0,
                step=1,
                key="prediction_pay_jul"
            )

            PAY_JUN = st.number_input(
                "PAY_JUN",
                value=0,
                step=1,
                key="prediction_pay_jun"
            )

        with col3:

            PAY_MAY = st.number_input(
                "PAY_MAY",
                value=0,
                step=1,
                key="prediction_pay_may"
            )

            PAY_APR = st.number_input(
                "PAY_APR",
                value=0,
                step=1,
                key="prediction_pay_apr"
            )

        # ----------------------------------------------------
        # BILL AMOUNTS
        # ----------------------------------------------------

        st.subheader("🧾 Bill Amounts")

        col1, col2, col3 = st.columns(3)

        with col1:

            BILL_AMT_SEPT = st.number_input(
                "BILL_AMT_SEPT",
                value=0.0,
                key="prediction_bill_sept"
            )

            BILL_AMT_AUG = st.number_input(
                "BILL_AMT_AUG",
                value=0.0,
                key="prediction_bill_aug"
            )

        with col2:

            BILL_AMT_JUL = st.number_input(
                "BILL_AMT_JUL",
                value=0.0,
                key="prediction_bill_jul"
            )

            BILL_AMT_JUN = st.number_input(
                "BILL_AMT_JUN",
                value=0.0,
                key="prediction_bill_jun"
            )

        with col3:

            BILL_AMT_MAY = st.number_input(
                "BILL_AMT_MAY",
                value=0.0,
                key="prediction_bill_may"
            )

            BILL_AMT_APR = st.number_input(
                "BILL_AMT_APR",
                value=0.0,
                key="prediction_bill_apr"
            )

        # ----------------------------------------------------
        # PAYMENT AMOUNTS
        # ----------------------------------------------------

        st.subheader("💰 Payment Amounts")

        col1, col2, col3 = st.columns(3)

        with col1:

            PAY_AMT_SEPT = st.number_input(
                "PAY_AMT_SEPT",
                value=0.0,
                key="prediction_pay_amt_sept"
            )

            PAY_AMT_AUG = st.number_input(
                "PAY_AMT_AUG",
                value=0.0,
                key="prediction_pay_amt_aug"
            )

        with col2:

            PAY_AMT_JUL = st.number_input(
                "PAY_AMT_JUL",
                value=0.0,
                key="prediction_pay_amt_jul"
            )

            PAY_AMT_JUN = st.number_input(
                "PAY_AMT_JUN",
                value=0.0,
                key="prediction_pay_amt_jun"
            )

        with col3:

            PAY_AMT_MAY = st.number_input(
                "PAY_AMT_MAY",
                value=0.0,
                key="prediction_pay_amt_may"
            )

            PAY_AMT_APR = st.number_input(
                "PAY_AMT_APR",
                value=0.0,
                key="prediction_pay_amt_apr"
            )

        # ----------------------------------------------------
        # PREDICT BUTTON
        # ----------------------------------------------------

        predict_button = st.form_submit_button(
            "🔮 Predict Risk"
        )

        # ----------------------------------------------------
        # API PREDICTION
        # ----------------------------------------------------

        if predict_button:

            customer_data = {

                "LIMIT_BAL": LIMIT_BAL,
                "SEX": SEX,
                "EDUCATION": EDUCATION,
                "MARRIAGE": MARRIAGE,
                "AGE": AGE,

                "PAY_SEPT": PAY_SEPT,
                "PAY_AUG": PAY_AUG,
                "PAY_JUL": PAY_JUL,
                "PAY_JUN": PAY_JUN,
                "PAY_MAY": PAY_MAY,
                "PAY_APR": PAY_APR,

                "BILL_AMT_SEPT": BILL_AMT_SEPT,
                "BILL_AMT_AUG": BILL_AMT_AUG,
                "BILL_AMT_JUL": BILL_AMT_JUL,
                "BILL_AMT_JUN": BILL_AMT_JUN,
                "BILL_AMT_MAY": BILL_AMT_MAY,
                "BILL_AMT_APR": BILL_AMT_APR,

                "PAY_AMT_SEPT": PAY_AMT_SEPT,
                "PAY_AMT_AUG": PAY_AMT_AUG,
                "PAY_AMT_JUL": PAY_AMT_JUL,
                "PAY_AMT_JUN": PAY_AMT_JUN,
                "PAY_AMT_MAY": PAY_AMT_MAY,
                "PAY_AMT_APR": PAY_AMT_APR
            }

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=customer_data,
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Prediction completed successfully!"
                    )

                    st.subheader(
                        "📊 Credit Risk Result"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Default Probability",
                            f"{result['default_probability']:.2%}"
                        )

                    with col2:

                        st.metric(
                            "Risk Category",
                            result["risk_category"]
                        )

                    with col3:

                        st.metric(
                            "Banking Decision",
                            result["banking_decision"]
                        )

                    # ------------------------------------------------
                    # RISK ASSESSMENT
                    # ------------------------------------------------

                    st.subheader("💡 Risk Assessment")

                    probability = result["default_probability"]

                    if probability >= 0.5:

                        st.error(
                            "🔴 High probability of default. "
                            "Detailed credit review is recommended."
                        )

                    elif probability >= 0.25:

                        st.warning(
                            "🟠 Moderate-to-high default probability. "
                            "Further review is recommended."
                        )

                    else:

                        st.success(
                            "🟢 Lower default probability. "
                            "Customer appears relatively lower risk."
                        )

                    # ------------------------------------------------
                    # COMPLETE API RESPONSE
                    # ------------------------------------------------

                    with st.expander(
                        "View Complete API Response"
                    ):

                        st.json(result)

                else:

                    st.error(
                        f"API returned HTTP {response.status_code}"
                    )

                    try:

                        st.json(response.json())

                    except Exception:

                        st.write(response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI server."
                )

                st.info(
                    "Make sure FastAPI is running on "
                    "http://127.0.0.1:8000"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ The FastAPI server took too long to respond."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ API request failed: {e}"
                )


# ============================================================
# PAGE 3 — EXPLAINABILITY
# ============================================================

elif page == "🔍 Explainability":

    st.header("🔍 Explainability")

    st.write(
        "Understand which customer features influence the "
        "credit risk prediction."
    )

    st.info(
        "Enter customer information below to generate a SHAP explanation."
    )

    # --------------------------------------------------------
    # LOAD FINAL MODEL
    # --------------------------------------------------------

    MODEL_PATH = "notebooks/final_credit_risk_model.pkl"
    FEATURES_PATH = "notebooks/final_features.pkl"

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        SHAP_LIMIT_BAL = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=0.0,
            value=20000.0,
            key="shap_limit_bal"
        )

    with col2:

        SHAP_SEX = st.number_input(
            "Gender (SEX)",
            min_value=1,
            max_value=2,
            value=2,
            step=1,
            key="shap_sex"
        )

    with col3:

        SHAP_EDUCATION = st.number_input(
            "Education (EDUCATION)",
            min_value=0,
            value=2,
            step=1,
            key="shap_education"
        )

    col1, col2 = st.columns(2)

    with col1:

        SHAP_MARRIAGE = st.number_input(
            "Marriage (MARRIAGE)",
            min_value=0,
            value=1,
            step=1,
            key="shap_marriage"
        )

    with col2:

        SHAP_AGE = st.number_input(
            "Age (AGE)",
            min_value=18,
            value=30,
            step=1,
            key="shap_age"
        )

    # --------------------------------------------------------
    # PAYMENT HISTORY
    # --------------------------------------------------------

    st.subheader("💳 Payment History")

    col1, col2, col3 = st.columns(3)

    with col1:

        SHAP_PAY_SEPT = st.number_input(
            "PAY_SEPT",
            value=0,
            step=1,
            key="shap_pay_sept"
        )

        SHAP_PAY_AUG = st.number_input(
            "PAY_AUG",
            value=0,
            step=1,
            key="shap_pay_aug"
        )

    with col2:

        SHAP_PAY_JUL = st.number_input(
            "PAY_JUL",
            value=0,
            step=1,
            key="shap_pay_jul"
        )

        SHAP_PAY_JUN = st.number_input(
            "PAY_JUN",
            value=0,
            step=1,
            key="shap_pay_jun"
        )

    with col3:

        SHAP_PAY_MAY = st.number_input(
            "PAY_MAY",
            value=0,
            step=1,
            key="shap_pay_may"
        )

        SHAP_PAY_APR = st.number_input(
            "PAY_APR",
            value=0,
            step=1,
            key="shap_pay_apr"
        )

    # --------------------------------------------------------
    # BILL AMOUNTS
    # --------------------------------------------------------

    st.subheader("🧾 Bill Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:

        SHAP_BILL_SEPT = st.number_input(
            "BILL_AMT_SEPT",
            value=0.0,
            key="shap_bill_sept"
        )

        SHAP_BILL_AUG = st.number_input(
            "BILL_AMT_AUG",
            value=0.0,
            key="shap_bill_aug"
        )

    with col2:

        SHAP_BILL_JUL = st.number_input(
            "BILL_AMT_JUL",
            value=0.0,
            key="shap_bill_jul"
        )

        SHAP_BILL_JUN = st.number_input(
            "BILL_AMT_JUN",
            value=0.0,
            key="shap_bill_jun"
        )

    with col3:

        SHAP_BILL_MAY = st.number_input(
            "BILL_AMT_MAY",
            value=0.0,
            key="shap_bill_may"
        )

        SHAP_BILL_APR = st.number_input(
            "BILL_AMT_APR",
            value=0.0,
            key="shap_bill_apr"
        )

    # --------------------------------------------------------
    # PAYMENT AMOUNTS
    # --------------------------------------------------------

    st.subheader("💰 Payment Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:

        SHAP_PAY_AMT_SEPT = st.number_input(
            "PAY_AMT_SEPT",
            value=0.0,
            key="shap_pay_amt_sept"
        )

        SHAP_PAY_AMT_AUG = st.number_input(
            "PAY_AMT_AUG",
            value=0.0,
            key="shap_pay_amt_aug"
        )

    with col2:

        SHAP_PAY_AMT_JUL = st.number_input(
            "PAY_AMT_JUL",
            value=0.0,
            key="shap_pay_amt_jul"
        )

        SHAP_PAY_AMT_JUN = st.number_input(
            "PAY_AMT_JUN",
            value=0.0,
            key="shap_pay_amt_jun"
        )

    with col3:

        SHAP_PAY_AMT_MAY = st.number_input(
            "PAY_AMT_MAY",
            value=0.0,
            key="shap_pay_amt_may"
        )

        SHAP_PAY_AMT_APR = st.number_input(
            "PAY_AMT_APR",
            value=0.0,
            key="shap_pay_amt_apr"
        )

    # --------------------------------------------------------
    # EXPLAIN BUTTON
    # --------------------------------------------------------

    explain_button = st.button(
        "🔍 Explain Prediction",
        key="shap_explain_button"
    )

    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------

    if explain_button:

        customer_values = [

            SHAP_LIMIT_BAL,
            SHAP_SEX,
            SHAP_EDUCATION,
            SHAP_MARRIAGE,
            SHAP_AGE,

            SHAP_PAY_SEPT,
            SHAP_PAY_AUG,
            SHAP_PAY_JUL,
            SHAP_PAY_JUN,
            SHAP_PAY_MAY,
            SHAP_PAY_APR,

            SHAP_BILL_SEPT,
            SHAP_BILL_AUG,
            SHAP_BILL_JUL,
            SHAP_BILL_JUN,
            SHAP_BILL_MAY,
            SHAP_BILL_APR,

            SHAP_PAY_AMT_SEPT,
            SHAP_PAY_AMT_AUG,
            SHAP_PAY_AMT_JUL,
            SHAP_PAY_AMT_JUN,
            SHAP_PAY_AMT_MAY,
            SHAP_PAY_AMT_APR
        ]

        customer_df = pd.DataFrame(
            [customer_values],
            columns=feature_names
        )

        try:

            # ------------------------------------------------
            # CREATE SHAP EXPLAINER
            # ------------------------------------------------

            explainer = shap.TreeExplainer(model)

            # ------------------------------------------------
            # CALCULATE SHAP VALUES
            # ------------------------------------------------

            shap_values = explainer.shap_values(
                customer_df
            )

            st.success(
                "✅ SHAP explanation generated successfully!"
            )

            # ------------------------------------------------
            # EXTRACT CLASS-1 SHAP VALUES
            # ------------------------------------------------

            if isinstance(shap_values, list):

                values = shap_values[1][0]

            else:

                shap_array = shap_values

                if len(shap_array.shape) == 3:

                    values = shap_array[0, :, 1]

                elif len(shap_array.shape) == 2:

                    values = shap_array[0]

                else:

                    values = shap_array

            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            st.subheader(
                "📊 Feature Importance"
            )

            importance_df = pd.DataFrame({

                "Feature": feature_names,

                "SHAP Value": values

            })

            importance_df["Absolute SHAP"] = (
                importance_df["SHAP Value"].abs()
            )

            importance_df = importance_df.sort_values(
                "Absolute SHAP",
                ascending=False
            )

            st.dataframe(

                importance_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].head(10),

                use_container_width=True,

                hide_index=True
            )

            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            st.subheader(
                "📈 SHAP Feature Impact"
            )

            plot_df = (
                importance_df
                .head(10)
                .sort_values(
                    "SHAP Value"
                )
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Value"]
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top Features Influencing Credit Risk"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        except Exception as e:

            st.error(
                f"❌ SHAP explanation failed: {e}"
            )


# ============================================================
# PAGE 4 — PRODUCT RECOMMENDATION
# ============================================================

# ============================================================
# PAGE 4 — PRODUCT RECOMMENDATION
# ============================================================

elif page == "🛍️ Product Recommendation":

    st.header("🛍️ Product Recommendation")

    st.write(
        "Recommend suitable banking products based on "
        "customer's existing products."
    )

    st.subheader("👤 Customer Selection")

    customer_ids = sorted(
        recommendation_df["Customer_ID"].unique()
    )

    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_ids,
        key="recommendation_customer_id"
    )

    # Get recommendations for selected customer
    customer_recommendations = recommendation_df[
        recommendation_df["Customer_ID"] == selected_customer
    ]

    if len(customer_recommendations) > 0:

        existing_products = (
            customer_recommendations["Existing_Products"]
            .iloc[0]
        )

        st.subheader("🏦 Existing Products")

        st.info(existing_products)

        st.subheader("🛍️ Recommended Products")

        for _, row in customer_recommendations.iterrows():

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(
                    f"🛍️ {row['Recommended_Product']}"
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{row['Confidence']:.2%}"
                )

            with col3:
                st.metric(
                    "Lift",
                    f"{row['Lift']:.2f}"
                )

        st.subheader("📊 Recommendation Details")

        display_df = customer_recommendations[
            [
                "Recommended_Product",
                "Confidence",
                "Lift",
                "Support"
            ]
        ].copy()

        display_df["Confidence"] = (
            display_df["Confidence"] * 100
        ).round(2).astype(str) + "%"

        display_df["Support"] = (
            display_df["Support"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True
        )

    else:

        st.warning(
            "No product recommendation is available "
            "for this customer."
        )
   