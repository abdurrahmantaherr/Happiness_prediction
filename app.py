import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset
df = pd.read_csv("2019.csv")

st.set_page_config(page_title="World Happiness Predictor", layout="wide")
st.title("🌍 World Happiness Report - Interactive Dashboard")

# ----------------------
# 📊 EDA SECTION
# ----------------------
st.header("📊 Exploratory Data Analysis")

if st.checkbox("Show raw data"):
    st.write(df)

if st.checkbox("Show summary statistics"):
    st.write(df.describe())

if st.checkbox("Show correlation heatmap"):
    st.subheader("Correlation Matrix")
    fig1, ax1 = plt.subplots()
    numeric_df = df.select_dtypes(include=['number'])  
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

if st.checkbox("GDP per Capita Distribution"):
    fig2, ax2 = plt.subplots()
    sns.histplot(df["GDP per capita"], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

if st.checkbox("Score vs Life Expectancy"):
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="Healthy life expectancy", y="Score", ax=ax3)
    st.pyplot(fig3)

st.markdown("---")

# ----------------------
# 🤖 PREDICTION SECTION
# ----------------------
st.header("🧠 Predict Happiness Score")

st.markdown("Enter the following values to get a predicted happiness score:")

gdp = st.number_input('GDP per capita', min_value=0.0, max_value=2.0, value=1.0)
social_support = st.number_input('Social support', min_value=0.0, max_value=1.5, value=1.0)
life_expectancy = st.number_input('Healthy life expectancy', min_value=0.0, max_value=1.2, value=0.8)
freedom = st.number_input('Freedom to make life choices', min_value=0.0, max_value=1.0, value=0.5)
generosity = st.number_input('Generosity', min_value=-0.5, max_value=0.5, value=0.1)
corruption = st.number_input('Perceptions of corruption', min_value=0.0, max_value=1.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[gdp, social_support, life_expectancy, freedom, generosity, corruption]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    st.success(f"Predicted Happiness Score: {prediction[0]:.2f}")

st.markdown("---")

# ----------------------
# 📝 CONCLUSION
# ----------------------
st.header("📌 Conclusion")

st.write("""
In this project, we analyzed the *World Happiness Report 2019* to understand what drives happiness across nations.
Using data science techniques — EDA, preprocessing, and machine learning — we trained a model that predicts happiness scores
based on key factors like GDP, social support, and life expectancy.

The model is deployed in this app, where users can explore the data and make real-time predictions. This showcases how data 
science can turn raw data into meaningful, interactive tools.
""")