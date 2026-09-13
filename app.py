import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
@st.cache_resource
def load_model(): return joblib.load("model.pkl")
@st.cache_data
def load_data(): return pd.read_csv("Telco-Customer-Churn.csv")
model=load_model()
df=load_data()
page=st.sidebar.radio("Navigation",["Business Problem","Data Insights","Prediction"])
if page=="Business Problem":
    st.title("Customer Churn Prediction Using Machine Learning")
    st.subheader("Business Problem")
    st.write("A telecom company wants to identify customers who are likely to leave so that retention actions can be taken before churn occurs.")
    st.subheader("Analytics Objective")
    st.write("Build a classification model that predicts whether a customer is likely to churn using demographic, service, contract and billing information.")
    st.subheader("Business Decision")
    st.write("Use the churn probability to prioritize retention campaigns, service improvements and customer engagement.")
    st.metric("Customers",f"{len(df):,}")
    st.metric("Observed Churn Rate",f"{(df['Churn']=='Yes').mean()*100:.1f}%")
elif page=="Data Insights":
    st.title("Data Insights")
    st.dataframe(df.head(10))
    st.subheader("Churn Distribution")
    st.bar_chart(df["Churn"].value_counts())
    st.subheader("Churn Rate by Contract")
    st.bar_chart(df.assign(ChurnFlag=(df["Churn"]=="Yes").astype(int)).groupby("Contract")["ChurnFlag"].mean())
    st.subheader("Churn Rate by Internet Service")
    st.bar_chart(df.assign(ChurnFlag=(df["Churn"]=="Yes").astype(int)).groupby("InternetService")["ChurnFlag"].mean())
    st.info("Customers on shorter contracts generally show higher churn. Tenure and service/contract characteristics are useful for retention prioritization.")
else:
    st.title("Predict Customer Churn")
    st.write("Enter customer details and estimate churn risk.")
    # defaults based on dataset
    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox("Gender",sorted(df.gender.unique()))
        senior=st.selectbox("Senior Citizen",[0,1])
        partner=st.selectbox("Partner",sorted(df.Partner.unique()))
        dependents=st.selectbox("Dependents",sorted(df.Dependents.unique()))
        tenure=st.number_input("Tenure (months)",0,72,12)
        phone=st.selectbox("Phone Service",sorted(df.PhoneService.unique()))
        multiple=st.selectbox("Multiple Lines",sorted(df.MultipleLines.unique()))
    with c2:
        internet=st.selectbox("Internet Service",sorted(df.InternetService.unique()))
        security=st.selectbox("Online Security",sorted(df.OnlineSecurity.unique()))
        backup=st.selectbox("Online Backup",sorted(df.OnlineBackup.unique()))
        device=st.selectbox("Device Protection",sorted(df.DeviceProtection.unique()))
        support=st.selectbox("Tech Support",sorted(df.TechSupport.unique()))
        tv=st.selectbox("Streaming TV",sorted(df.StreamingTV.unique()))
        movies=st.selectbox("Streaming Movies",sorted(df.StreamingMovies.unique()))
    with c3:
        contract=st.selectbox("Contract",sorted(df.Contract.unique()))
        paperless=st.selectbox("Paperless Billing",sorted(df.PaperlessBilling.unique()))
        payment=st.selectbox("Payment Method",sorted(df.PaymentMethod.unique()))
        monthly=st.number_input("Monthly Charges",0.0,200.0,70.0)
        total=st.number_input("Total Charges",0.0,10000.0,monthly*tenure)
    row=pd.DataFrame([{"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}])
    if st.button("Predict Churn Risk"):
        prob=model.predict_proba(row)[0,1]
        label="High Churn Risk" if prob>=0.5 else "Low Churn Risk"
        st.metric("Predicted Churn Probability",f"{prob*100:.1f}%")
        if prob>=0.5: st.error(label+" — prioritize this customer for retention.")
        else: st.success(label+" — standard engagement may be sufficient.")
