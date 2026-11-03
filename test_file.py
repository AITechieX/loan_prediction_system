import streamlit as st
import joblib
import numpy as np
from groq import Groq
import os
from datetime import datetime
import pandas as pd
import json
from config import Config


class LoanPredictor:
    def __init__(self):
        self.model = joblib.load('random_forest_model_chatgpt_dataset.pkl')
        self.scaler = joblib.load('scaler_chatgpt_dataset.pkl')
        self.client = Groq(api_key=Config.GROQ_API_KEY)

    def analyze(self, features):
        scaled_features = self.scaler.transform(features)
        prediction = self.model.predict(scaled_features)
        return prediction[0], self._get_analysis(features[0], prediction[0])

    def _get_analysis(self, features, is_approved):
        status = "APPROVED" if is_approved else "NOT APPROVED"
        prompt = f"""Analyze this loan application ({status}):
        Credit Score: {features[0]}
        Income: ${features[1]:,.2f}
        DTI: {features[2]:.2%}
        Loan Amount: ${features[3]:,.2f}
        Interest: {features[4]:.1f}%
        Term: {features[5]} years
        Employment: {['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired', 'Other'][int(features[6])]}
        Age: {features[7]}
        Assets: ${features[8]:,.2f}
        Liabilities: ${features[9]:,.2f}

        Provide expert financial analysis with:
        1. Key decision factors
        2. Risk assessment
        3. Recommendations
        4. Financial metrics analysis
        """

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.GROQ_MODEL_TEST,
                temperature=Config.GROQ_TEMPERATURE,
                max_tokens=Config.GROQ_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"


st.set_page_config(layout="wide", page_title="Loan Prediction System", page_icon="🏦")

# [Previous CSS styles remain unchanged]
st.markdown("""
<style>
    /* All your existing CSS styles here */
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown(
    '<div class="main-header"><h1 class="title">Loan Prediction System</h1><p class="subtitle">AI-Powered Financial Analysis</p></div>',
    unsafe_allow_html=True)

predictor = LoanPredictor()

# Input Form
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h3 class="section-title">Financial Information</h3>', unsafe_allow_html=True)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
        income = st.number_input('Income ($)', min_value=0, value=50000, step=1000)
        debt_to_income_ratio = st.number_input('Debt-to-Income Ratio', min_value=0.0, max_value=1.0, value=0.2)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3 class="section-title">Loan Details</h3>', unsafe_allow_html=True)
        loan_amount = st.number_input('Loan Amount ($)', min_value=0, value=100000, step=1000)
        interest_rate = st.number_input('Interest Rate (%)', min_value=0.0, value=5.0, step=0.1)
        repayment_term = st.number_input('Repayment Term (Years)', min_value=1, value=5, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><h3 class="section-title">Personal Details</h3>', unsafe_allow_html=True)
        employment_status = st.selectbox('Employment Status',
                                         ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired', 'Other'])
        age = st.number_input('Age (Years)', min_value=18, value=30, step=1)
        asset_value = st.number_input('Asset Value ($)', min_value=0, value=500000, step=1000)
        liabilities = st.number_input('Liabilities ($)', min_value=0, value=100000, step=1000)
        st.markdown('</div>', unsafe_allow_html=True)

if st.button("ANALYZE APPLICATION", use_container_width=True):
    employment_status_encoded = ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired', 'Other'].index(employment_status)

    features = np.array([[credit_score, income, debt_to_income_ratio, loan_amount,
                         interest_rate, repayment_term, employment_status_encoded,
                         age, asset_value, liabilities]])

    prediction, analysis = predictor.analyze(features)

    # Display Result
    if prediction == 1:
        st.markdown('<div class="result-approved">LOAN APPROVED ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-rejected">LOAN NOT APPROVED ❌</div>', unsafe_allow_html=True)

    # Key Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.markdown(f"""
           <div class="metric-container">
               <div class="metric-value">{debt_to_income_ratio:.1%}</div>
               <div class="metric-label">Debt-to-Income Ratio</div>
           </div>
           """, unsafe_allow_html=True)

    with metrics_col2:
        ltv = loan_amount / asset_value if asset_value > 0 else 0
        st.markdown(f"""
           <div class="metric-container">
               <div class="metric-value">{ltv:.1%}</div>
               <div class="metric-label">Loan-to-Value Ratio</div>
           </div>
           """, unsafe_allow_html=True)

    with metrics_col3:
        monthly_payment = (loan_amount * (interest_rate / 1200) * (1 + interest_rate / 1200) ** (
                    repayment_term * 12)) / (
                                  (1 + interest_rate / 1200) ** (repayment_term * 12) - 1)
        st.markdown(f"""
           <div class="metric-container">
               <div class="metric-value">{monthly_payment:,.0f}</div>
               <div class="metric-label">Est. Monthly Payment</div>
           </div>
           """, unsafe_allow_html=True)

    with metrics_col4:
        debt_coverage = income / (monthly_payment * 12) if monthly_payment > 0 else 0
        st.markdown(f"""
           <div class="metric-container">
               <div class="metric-value">{debt_coverage:.1f}x</div>
               <div class="metric-label">Debt Coverage Ratio</div>
           </div>
           """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # AI Analysis
    with st.expander("📊 View Detailed AI Analysis", expanded=True):
        st.markdown(f"<div class='card'>{analysis}</div>", unsafe_allow_html=True)
