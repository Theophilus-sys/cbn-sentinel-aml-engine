import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import joblib
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from fpdf import FPDF

# Setting Up Assets
st.set_page_config(page_title="The Sentinel: AML Compliance Engine", layout="wide")
load_dotenv()
#Secure API Key for Deployment 
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Groq API Key not found. AML Sentinel offline😏")

@st.cache_resource
def load_ml_assets():
    model = joblib.load("AntiMoneyLaundering_Model.pkl")
    scaler = joblib.load("AML_scaler.pkl")
    encoder = joblib.load("aml_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_ml_assets()
#For the conversion of .txt file into pdf
def generate_sar_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = text.replace("₦","NGN").replace("\u20A6","NGN")
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output(dest="S").encode('latin-1')
# LLM SAR Prompt Drafting
llm = ChatGroq(temperature = 0.1,model_name="llama-3.3-70b-versatile",groq_api_key = api_key)
template_text = """
    ROLE: Senior AML Compliance Officer, Central Bank of Nigeria (CBN).
    TASK: Draft a formal Suspicious Activity Report (SAR) for the NFIU.
    
    DATA EVIDENCE:
    - Date: (current_date)
    - Subject: {subject}
    - Account Tier: {account_tier}
    - Total Volume: NGN {total_volume}
    - Transaction Count: {transaction_freq} in 24 hours
    - Avg Transaction: NGN {avg_transaction}
    - Location: {location}
    - Violation: {violation}
    
    LEGAL REQUIREMENT: 
    Reference the 'Money Laundering (Prevention and Prohibition) Act 2022'. 
    Explain why this behavior constitutes {violation}.
    Recommend a 'Post-No-Debit' (PND) status.
    """
# Integrating the llm with the prompt
sar_prompt = PromptTemplate.from_template(template_text)
sar_chain = sar_prompt | llm

#Sidebar and Frontpage Customization
st.title("🛡️ AML Sentinel Compliance Engine")
st.markdown("An effort at enforcing compliance to CBN policy")
st.sidebar.title("Data Holder")
st.sidebar.caption("Upload your dataset here")
st.sidebar.info("files received are csv or excel and not both")
uploaded_file = st.sidebar.file_uploader("Dataset uploading", type=["csv","xlsx"])
#Default Dataset provided 
default_dataset = "cbn_sentinel_data1.csv"
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("Data successfully uploaded")
elif os.path.exists(default_dataset):
    df = pd.read_csv(default_dataset)
    st.sidebar.info("Using default dataset")
else:
    st.info("No file found")
    
# Feature Engineerin g for Higher level model's performance
if df is not None:
    amount_column = "Amount(\u20A6)"
    df["Txn_freq_24h"] = df.groupby("Customer ID")[amount_column].transform("count")
    df["Txn_volume"] = df.groupby("Customer ID")[amount_column].transform("sum")
    df["Avg_Txn_Size"] = df["Txn_volume"] / df["Txn_freq_24h"]
    pd.options.display.float_format = "{:,.2f}".format #this tells pandas to show numbers in 2 decimal places instead of scientific notation
    df["Round Number"] = df[amount_column].apply(lambda x:1 if  x % 1000 == 0 else 0)
    location_risk = {
        "Enugu": 4,  #High Cybercrime Zone
        "Ibadan": 4, #High Cybercrime Zone
        "Port Harcourt": 3, #High Cybercrime Zone
        "Abuja":3, #High Political Zone
        "Lagos":5 #Highest risk Zone
    }
    df["High_Risk_Location"] = df["Sender's Location"].map(location_risk).fillna(0)

numeric_features = [amount_column,"Txn_freq_24h","Txn_volume","Avg_Txn_Size","Round Number","High_Risk_Location"]
categorical_features = ["Transaction Type","Account Type","Account Tier"]
encoded_categories = encoder.transform(df[categorical_features])
if hasattr(encoded_categories,"toarray"):
    encoded_categories = encoded_categories.toarray()

X_combine = np.hstack([df[numeric_features].values, encoded_categories])
scaled_data = scaler.transform(X_combine)
modeler = model.predict(scaled_data)
df["Risk_Status"] =["Flagged" if x == -1 else "Clear" for x in modeler]

#Dashboard Visual
st.write("### Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.subheader("📊Sentinel Intelligence Map")
fig = px.scatter(
df,
x="Txn_volume",#for showing structuring
y="Txn_freq_24h",#For showing velocity
color="Risk_Status",
color_discrete_map={"Flagged": "#EF553B","Clear":"#00CC96"},
hover_data=[amount_column,"Account Tier","Transaction Type"],
title="Sentinel Intelligence: Transaction Risk Map",
template="plotly_dark"
)
st.plotly_chart(fig,use_container_width=True)
flag_df = df[df["Risk_Status"] == "Flagged"]

if not flag_df.empty:
    st.subheader("Flagged Transactions")
    st.dataframe(flag_df, use_container_width=True)
    st.divider()
    st.subheader("Automated SAR Narrative")
    selected_idx = st.selectbox("Select Transaction Index to report from the dropdown menu", flag_df.index)
    record = flag_df.loc[selected_idx]
    if "sar_report" not in st.session_state:
        st.session_state.sar_report = None
    if "last_index" not in st.session_state or st.session_state.last_index != selected_idx:
        st.session_state.sar_report = None
        st.session_state.last_index = selected_idx
    if st.button("Generate SAR Draft"):
        with st.spinner("Drafting CBN_Compliant Report..."):
            current_date = datetime.now().strftime("%B %d, %Y"),
            txn_freq =record.get("Txn_freq_24h", 0)
            violation = "Structuring/Smurfing" if txn_freq > 10 else "Threshold Breach"
            record_input = { 
                "subject":record.get('Customer ID'),
                "account_tier":record.get('Account Tier'),
                "total_volume":f"{record.get('Txn_volume',0):,.2f}",
                "transaction_freq":txn_freq,
                "avg_transaction":f"{record.get('Avg_Txn_Size',0):,.2f}",
                "location":record.get("Sender's Location"),
                "violation":violation
            }
            response = sar_chain.invoke(record_input)
            st.session_state.sar_report=response.content

    if st.session_state.sar_report:
        st.info(st.session_state.sar_report)
        pdf_bytes = generate_sar_pdf(st.session_state.sar_report)
        st.download_button(
            label="Download Official SAR (.PDF)",
            data=pdf_bytes,
            file_name=f"SAR_{selected_idx}.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="Download Official SAR (.txt)",
            data=st.session_state.sar_report,
            file_name=f"SAR_{selected_idx}.txt"
        )
    
else:
    st.success("Clean Sweep: No anomalies detected")
