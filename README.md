# Banking Credit Risk & Cross-Sell Recommendation System

## 📌 Project Overview

The **Banking Credit Risk & Cross-Sell Recommendation System** is an end-to-end machine learning application designed to help banks identify customers who are at higher risk of credit default and recommend suitable banking products.

The system combines **credit risk prediction, explainable AI, association rule mining, product recommendation, REST API services, and an interactive dashboard** into a single integrated banking analytics solution.

## 🎯 Objectives

* Predict the probability of customer credit default.
* Classify customers into risk categories.
* Provide banking decisions based on predicted risk.
* Explain the factors influencing credit-risk predictions.
* Discover relationships between banking products.
* Recommend suitable products to customers.
* Provide predictions through a REST API.
* Provide an interactive dashboard for users.

## 🚀 Key Features

### 1. Credit Risk Prediction

Machine learning models are used to predict the probability of credit default.

### 2. Risk Classification

Customers are categorized into:

* Low Risk
* High Risk

The system also generates a banking decision based on the predicted risk.

### 3. Explainable AI

SHAP-based explainability is used to understand which customer features contribute to the model's prediction.

### 4. Cross-Sell Recommendation

Association rule mining is used to identify relationships between banking products and generate product recommendations.

### 5. REST API

A FastAPI backend exposes the prediction functionality through HTTP endpoints.

### 6. Interactive Dashboard

A Streamlit dashboard provides:

* 🏠 Banking dashboard
* 🔮 Credit risk prediction
* 🔍 SHAP-based risk explanation
* 🛍️ Product recommendations

## 🧠 Machine Learning

The project uses multiple machine learning techniques, including:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

The final system uses the trained model artifacts stored in the `models/` directory.

## 📊 Dataset

The project is based on the **UCI Default of Credit Card Clients** dataset.

The dataset contains customer demographic information, credit-limit information, repayment-status information, billing amounts, payment amounts, and the default target variable.

Feature engineering was performed to derive additional customer-level risk indicators such as:

* Average Bill Amount
* Average Payment Amount
* Credit Utilization
* Payment-to-Bill Ratio
* Payment Delay
* Severe Delay Indicators
* Recent Payment Delay
* Payment Delay Trend
* Credit Limit Category
* Age Category

## 🛍️ Recommendation System

Association Rule Mining is used to identify relationships among banking products such as:

* Savings Account
* Credit Card
* Debit Card
* Insurance
* Home Loan
* Personal Loan

Recommendations are generated using association-rule metrics such as:

* Support
* Confidence
* Lift

## 🏗️ Project Structure

```text
BANKING-CREDIT-RISK/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── credit_risk_model.pkl
│   ├── credit_risk_xgb_model.pkl
│   ├── feature_names.pkl
│   └── threshold.pkl
│
├── notebooks/
│   ├── 01_dataset_check.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_association_rules.ipynb
│   ├── 05_final_banking_system.ipynb
│   └── 11_fastapi.ipynb
│
├── src/
│   ├── explain.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── recommend.py
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies Used

### Programming

* Python 3.12.7

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SciPy
* Joblib

### Explainable AI

* SHAP

### Recommendation System

* MLxtend
* Association Rule Mining

### Visualization

* Matplotlib
* Seaborn

### Backend

* FastAPI
* Uvicorn

### Frontend / Dashboard

* Streamlit

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd BANKING-CREDIT-RISK
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## ▶️ Running the FastAPI Backend

From the project root:

```powershell
uvicorn api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 📊 Running the Streamlit Dashboard

Open another PowerShell terminal, activate the virtual environment, and run:

```powershell
streamlit run dashboard/app.py
```

The Streamlit dashboard will open in your browser.

## 🔌 API

The FastAPI backend provides endpoints for the banking credit-risk system.

The interactive API documentation can be accessed through:

```text
/docs
```

The API accepts customer information and returns the corresponding credit-risk prediction and banking decision.

## 📈 Model Evaluation

The project evaluates multiple classification models using metrics including:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Threshold tuning was also performed to improve the identification of customers at risk of default.

## 🔐 Project Security

Sensitive configuration files such as `.env` are excluded through `.gitignore`.

The virtual environment and Python cache files are also excluded from version control.

## 📚 Project Components

* Exploratory Data Analysis
* Data Preprocessing
* Credit Risk Prediction
* Machine Learning Model Training
* Explainable AI
* Association Rule Mining
* Cross-Sell Recommendation
* REST API
* Interactive Dashboard

## 👩‍💻 Project Status

**Status: Completed**

The project integrates the complete pipeline from data analysis and machine learning to API deployment and interactive dashboard visualization.
