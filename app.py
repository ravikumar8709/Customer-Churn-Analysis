import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load saved model and features
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Page config
st.set_page_config(page_title="Churn Prediction Dashboard",
                   page_icon="📊", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Predict Customer", "Model Insights"])

# ================================
# PAGE 1 - OVERVIEW
# ================================
if page == "Overview":
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("**Proactive retention analytics for credit card customers**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "10,127")
    col2.metric("Churned Customers", "1,627")
    col3.metric("Churn Rate", "16.07%")
    col4.metric("Model AUC-ROC", "98.91%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        labels = ['Retained (84%)', 'Churned (16%)']
        sizes = [8500, 1627]
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        st.subheader("Prioritization Tiers")
        fig, ax = plt.subplots(figsize=(5,3))
        tiers = ['High Risk', 'Medium Risk', 'Low Risk']
        counts = [295, 69, 1662]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        ax.bar(tiers, counts, color=colors)
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Risk Segments')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top Behavioral Precursors of Churn")
    precursors = pd.DataFrame({
        'Feature': ['Total_Trans_Ct', 'Inactivity_Rate', 'Total_Trans_Amt',
                    'Months_Inactive_12_mon', 'Total_Revolving_Bal'],
        'Importance': [4.4, 2.2, 1.4, 1.2, 1.0]
    })
    fig, ax = plt.subplots(figsize=(8,3))
    ax.barh(precursors['Feature'], precursors['Importance'], color='#3498db')
    ax.set_xlabel('SHAP Value (Impact on Churn)')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# ================================
# PAGE 2 - PREDICT CUSTOMER
# ================================
elif page == "Predict Customer":
    st.title("Predict Customer Churn Risk")
    st.markdown("Enter customer details to get churn probability and risk tier")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Customer Age", 18, 75, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent_count = st.slider("Dependent Count", 0, 5, 2)
        education = st.selectbox("Education Level",
            ["Graduate", "High School", "Unknown",
             "Uneducated", "College", "Post-Graduate", "Doctorate"])
        marital = st.selectbox("Marital Status",
            ["Married", "Single", "Unknown"])

    with col2:
        st.subheader("Financial Info")
        income = st.selectbox("Income Category",
            ["Less than $40K", "$40K - $60K", "$60K - $80K",
             "$80K - $120K", "$120K +", "Unknown"])
        card = st.selectbox("Card Category",
            ["Blue", "Silver", "Gold", "Platinum"])
        credit_limit = st.number_input("Credit Limit", 1000, 35000, 5000)
        revolving_bal = st.number_input("Total Revolving Balance", 0, 3000, 500)
        months_on_book = st.slider("Months on Book", 12, 56, 36)

    with col3:
        st.subheader("Activity Info")
        total_trans_ct = st.slider("Total Transactions (12 mon)", 10, 140, 60)
        total_trans_amt = st.number_input("Total Transaction Amount", 500, 20000, 4000)
        months_inactive = st.slider("Months Inactive (12 mon)", 0, 6, 2)
        contacts_count = st.slider("Contacts Count (12 mon)", 0, 6, 2)
        total_rel_count = st.slider("Total Relationship Count", 1, 6, 3)

    if st.button("Predict Churn Risk", type="primary"):

        # Encode inputs
        le = LabelEncoder()

        gender_enc = 0 if gender == "F" else 1
        edu_map = {"College":0,"Doctorate":1,"Graduate":2,
                   "High School":3,"Post-Graduate":4,"Uneducated":5,"Unknown":6}
        mar_map = {"Divorced":0,"Married":1,"Single":2,"Unknown":3}
        inc_map = {"$120K +":0,"$40K - $60K":1,"$60K - $80K":2,
                   "$80K - $120K":3,"Less than $40K":4,"Unknown":5}
        card_map = {"Blue":0,"Gold":1,"Platinum":2,"Silver":3}

        # Build input row matching training features
        input_data = {
            'Customer_Age': age,
            'Gender': gender_enc,
            'Dependent_count': dependent_count,
            'Education_Level': edu_map.get(education, 2),
            'Marital_Status': mar_map.get(marital, 1),
            'Income_Category': inc_map.get(income, 4),
            'Card_Category': card_map.get(card, 0),
            'Months_on_book': months_on_book,
            'Total_Relationship_Count': total_rel_count,
            'Months_Inactive_12_mon': months_inactive,
            'Contacts_Count_12_mon': contacts_count,
            'Credit_Limit': credit_limit,
            'Total_Revolving_Bal': revolving_bal,
            'Total_Amt_Chng_Q4_Q1': 0.8,
            'Total_Trans_Amt': total_trans_amt,
            'Total_Trans_Ct': total_trans_ct,
            'Total_Ct_Chng_Q4_Q1': 0.7,
            'Inactivity_Rate': months_inactive / 12,
            'Avg_Txn_Per_Month': total_trans_ct / months_on_book,
            'Avg_Txn_Amount': total_trans_amt / (total_trans_ct + 1),
            'Balance_Utilization': revolving_bal / (credit_limit + 1)
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        col1.metric("Churn Probability", f"{prob*100:.1f}%")

        if prob >= 0.7:
            tier = "🔴 HIGH RISK"
            col2.metric("Risk Tier", tier)
            col3.metric("Recommended Action", "Call Immediately")
            st.error(f"This customer has {prob*100:.1f}% churn probability — HIGH RISK")
        elif prob >= 0.3:
            tier = "🟡 MEDIUM RISK"
            col2.metric("Risk Tier", tier)
            col3.metric("Recommended Action", "Send Retention Offer")
            st.warning(f"This customer has {prob*100:.1f}% churn probability — MEDIUM RISK")
        else:
            tier = "🟢 LOW RISK"
            col2.metric("Risk Tier", tier)
            col3.metric("Recommended Action", "Monitor Only")
            st.success(f"This customer has {prob*100:.1f}% churn probability — LOW RISK")

# ================================
# PAGE 3 - MODEL INSIGHTS
# ================================
elif page == "Model Insights":
    st.title("Model Performance Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Comparison")
        models_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'AUC-ROC': [0.7165, 0.9826, 0.9891]
        })
        fig, ax = plt.subplots(figsize=(5,3))
        colors = ['#95a5a6', '#3498db', '#2ecc71']
        ax.bar(models_df['Model'], models_df['AUC-ROC'], color=colors)
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel('AUC-ROC Score')
        ax.set_title('Model Comparison')
        for i, v in enumerate(models_df['AUC-ROC']):
            ax.text(i, v + 0.005, str(v), ha='center', fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Confusion Matrix")
        cm = np.array([[1670, 31], [41, 284]])
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Retained','Churned'],
                    yticklabels=['Retained','Churned'], ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('XGBoost Confusion Matrix')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Model Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "96%")
    col2.metric("Precision", "90%")
    col3.metric("Recall", "87%")
    col4.metric("F1-Score", "89%")