# Customer Churn Prediction Using Machine Learning

## Business Problem
A telecom company wants to identify customers who are likely to leave so retention actions can be taken before churn occurs.

## Objective
Build and deploy a classification model that predicts customer churn using demographic, service, contract and billing information.

## Dataset
- Source: Kaggle Telco Customer Churn Dataset
- Rows after cleaning: 7032
- Target: Churn
- Task: Binary Classification

## Methodology
Data loading → cleaning → EDA → preprocessing → train/test split → Logistic Regression and Random Forest → evaluation → model saving → Streamlit deployment.

## Model Performance
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.805 | 0.652 | 0.575 | 0.611 | 0.836 |
| Random Forest | 0.785 | 0.625 | 0.476 | 0.540 | 0.812 |

## Final Model
Logistic Regression was selected because it achieved the better ROC-AUC, accuracy, recall and F1 score on the test set.

## Business Insights
Contract type, tenure, internet service and customer service characteristics are useful for identifying churn risk. Shorter-contract customers generally show higher churn.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure
```text
telco_churn_project/
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── model_development.ipynb
└── images/
```

## Deployment
Deploy `app.py` using Streamlit Community Cloud after pushing the repository to GitHub.

GitHub Repository: Add your GitHub link after creating the repository.
Live Streamlit Application: Add your deployment link after deployment.
