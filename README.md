# banking-credit-risk
# Banking Credit Risk & Cross-Sell Recommendation System

## 📌 Project Overview

This project is an end-to-end **Banking Credit Risk Prediction and Cross-Sell Recommendation System**.

The system predicts the probability of a customer defaulting on their credit payment, categorizes customers based on risk, provides banking decisions, and recommends suitable banking products using association rule mining.

The project was developed and completed through **13 phases**, covering data analysis, machine learning, feature engineering, cross-selling, API development, and dashboard integration.

---

# 🚀 Project Phases

## Phase 1 — Project Setup
- Project structure created
- Dataset and project requirements identified
- Development environment configured

## Phase 2 — Dataset Preparation
- UCI Credit Card Default dataset collected
- Data cleaning performed
- Missing values and duplicate records handled
- Dataset prepared for analysis

## Phase 3 — Exploratory Data Analysis
- Customer demographics analyzed
- Credit limit and billing behavior analyzed
- Payment behavior analyzed
- Default distribution investigated
- Important patterns identified

## Phase 4 — Feature Engineering
Created meaningful financial and behavioral features including:
- Average Bill Amount
- Average Payment Amount
- Total Bill Amount
- Total Payment Amount
- Credit Utilization
- Payment-to-Bill Ratio
- Maximum Bill Amount
- Payment Delay Features
- Severe Delay Indicators
- Recent Payment Behavior
- Age Category
- Credit Limit Category

## Phase 5 — Machine Learning Model Training
Multiple classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Phase 6 — Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

XGBoost provided the strongest overall performance.

## Phase 7 — Threshold Optimization
The classification threshold was optimized for better detection of customers likely to default.

**Final classification threshold: 0.25**

This improved the model's ability to identify higher-risk customers.

## Phase 8 — Final Credit Risk Model
The final credit risk pipeline was created using:
- Feature engineering
- Standard scaling
- XGBoost
- Risk probability prediction
- Risk categorization
- Banking decision logic

Customers are categorized into:
- Low Risk
- High Risk

## Phase 9 — Cross-Sell Recommendation System
Association rule mining was implemented to recommend banking products.

Products considered include:
- Savings Account
- Credit Card
- Debit Card
- Insurance
- Home Loan
- Personal Loan

Recommendations use:
- Support
- Confidence
- Lift

## Phase 10 — Banking System Integration
The credit risk model and cross-sell recommendation system were integrated into a unified banking analysis workflow.

The system produces:
- Customer ID
- Default Probability
- Risk Category
- Banking Decision
- Recommended Product
- Confidence
- Lift
- Support

## Phase 11 — API Development
A backend API was developed using **FastAPI**.

The API provides:
- Health check
- Credit risk prediction
- Banking decision
- Customer analysis

Swagger/OpenAPI documentation was also integrated for API testing.

## Phase 12 — Dashboard Development
A banking dashboard was developed to provide an interactive interface for the credit risk and cross-sell system.

The dashboard allows users to interact with the banking analysis system and view customer risk information and recommendations.

## Phase 13 — Final Verification & Integration
The complete system was tested and integrated.

Verified components include:
- Machine learning model
- Feature pipeline
- Credit risk prediction
- Risk categorization
- Banking decision
- Cross-sell recommendations
- FastAPI backend
- Dashboard
- API documentation

---

# 🏦 Final System Workflow

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Credit Risk Model
      ↓
Default Probability
      ↓
Risk Category
      ↓
Banking Decision
      ↓
Cross-Sell Recommendation
      ↓
FastAPI / Dashboard
