from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import joblib



st.set_page_config(page_title="Medical Insurance Prediction",
                   page_icon="💊",
                   layout="wide")

# Load model
model = joblib.load("C:/Users/arkha/jupyter-workspace/medical_insurance_cost_prediction_project/models/PolynomialRegressionPipeline.pkl")

# Load data for EDA
df = pd.read_csv("C:/Users/arkha/jupyter-workspace/medical_insurance_cost_prediction_project/data/cleaned_data.csv")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Predictions", "About Us"])



df = pd.read_csv('C:/Users/arkha/jupyter-workspace/medical_insurance_cost_prediction_project/data/cleaned_data.csv')
dfRaw = pd.read_csv('C:/Users/arkha/jupyter-workspace/medical_insurance_cost_prediction_project/data/medical_insurance.csv')

plt.rcParams['figure.figsize'] = (8,5)

# --- Page 1: Introduction ---
if page == "Introduction":
    st.title("💊 Medical Insurance Cost Prediction")
    st.markdown("""
    Welcome to **Medical Insurance Cost Prediction App**!  
    This tool helps you:
    - 🔍 Explore the dataset
    - 📊 Analyze trends
    - 🤖 Get predictions using ML models
    - 👨‍👩‍👧 Learn about the team
    """)


# --- Page 2: EDA ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    analysis_type = st.selectbox("🔎 Select Analysis Type", 
                                 ["Univariate", "Bivariate", "Multivariate", "Outliers", "Correlation"])

    # --- Univariate ---
    if analysis_type == "Univariate":
        with st.expander("📈 Explore Univariate Distributions", expanded=True):
            question = st.selectbox("Select Question", [
                "Distribution of Charges",
                "Age Distribution",
                "Smokers vs Non-Smokers",
                "BMI Distribution",
                "Number of Policyholders by Region"
            ])
            fig, ax = plt.subplots()
            if question == "Distribution of Charges":
                sns.histplot(df['charges'], kde=True, bins=30, color='skyblue', ax=ax)
                ax.set_xlabel("Charges"); ax.set_ylabel("Count")
                st.caption("Charges follow a skewed distribution with some high-cost outliers.")
            elif question == "Age Distribution":
                sns.histplot(df['age'], kde=True, bins=20, color='pink', ax=ax)
                ax.set_xlabel("Age"); ax.set_ylabel("Count")
                st.caption("Most policyholders are between 20 and 60 years old.")
            elif question == "Smokers vs Non-Smokers":
                sns.countplot(x='smoker', data=df, palette='pastel', ax=ax)
                ax.set_xlabel("Smoker (1=Yes, 0=No)"); ax.set_ylabel("Count")
                st.caption("Majority of policyholders are non-smokers.")
            elif question == "BMI Distribution":
                avg_bmi = df['bmi'].mean()
                sns.histplot(df['bmi'], kde=True, bins=20, color='orange', ax=ax)
                ax.axvline(avg_bmi, color='blue', linestyle='--', label=f'Average BMI={avg_bmi:.2f}')
                ax.set_xlabel("BMI"); ax.set_ylabel("Count"); ax.legend()
                st.caption("BMI clusters around normal/overweight range.")
            elif question == "Number of Policyholders by Region":
                sns.countplot(data=dfRaw, x='region', order=dfRaw['region'].value_counts().index, palette='Set2', ax=ax)
                ax.set_xlabel("Region"); ax.set_ylabel("Count")
                st.caption("South-east and south-west regions have higher representation.")
            st.pyplot(fig)

    # --- Bivariate ---
    elif analysis_type == "Bivariate":
        with st.expander("📊 Compare Two Variables", expanded=True):
            question = st.selectbox("Select Question", [
                "Charges vs Age",
                "Avg Charges: Smokers vs Non-Smokers",
                "BMI vs Charges",
                "Avg Charges: Men vs Women",
                "Avg Charges by Number of Children"
            ])
            fig, ax = plt.subplots()
            if question == "Charges vs Age":
                sns.scatterplot(x='age', y='charges', data=df, alpha=0.6, ax=ax)
                ax.set_xlabel("Age"); ax.set_ylabel("Charges")
                st.caption("Charges tend to increase with age, but with high variability.")
            elif question == "Avg Charges: Smokers vs Non-Smokers":
                sns.barplot(data=df, x='smoker', y='charges', palette='Set2', ci=None, ax=ax)
                ax.set_xlabel("Smoker [1=Yes | 0=No]"); ax.set_ylabel("Charges")
                st.caption("Smokers are charged significantly higher on average.")
            elif question == "BMI vs Charges":
                sns.scatterplot(data=df, x='bmi', y='charges', alpha=0.6, ax=ax)
                ax.set_xlabel("BMI"); ax.set_ylabel("Charges")
                st.caption("High BMI values often correspond with higher charges.")
            elif question == "Avg Charges: Men vs Women":
                sns.barplot(x='sex', y='charges', data=df, ci=None, palette='pastel', ax=ax)
                ax.set_xlabel("Sex"); ax.set_ylabel("Average Charges")
                st.caption("Average charges are similar across gender.")
            elif question == "Avg Charges by Number of Children":
                sns.barplot(data=df, x='children', y='charges', ci=None, palette='muted', ax=ax)
                ax.set_xlabel("#Children"); ax.set_ylabel("Average Charges")
                st.caption("Having children has only a small effect on charges.")
            st.pyplot(fig)

    # --- Multivariate ---
    elif analysis_type == "Multivariate":
        with st.expander("🔗 Multivariate Relationships", expanded=True):
            question = st.selectbox("Select Question", [
                "Charges vs Age by Smoking Status",
                "Avg Charges by Gender and Region (Smokers Only)",
                "Age, BMI and Smoking Status Impact on Charges",
                "Obese Smokers vs Non-Obese Non-Smokers"
            ])
            fig, ax = plt.subplots()
            if question == "Charges vs Age by Smoking Status":
                sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.6, palette='Set1', ax=ax)
                ax.set_xlabel("Age"); ax.set_ylabel("Charges"); ax.legend(title='Smoker', labels=['Yes','No'])
                st.caption("Smokers show higher charges across all ages.")
            elif question == "Avg Charges by Gender and Region (Smokers Only)":
                sns.barplot(data=dfRaw, x='sex', y='charges', hue='region', ci=None, palette='Set2', ax=ax)
                ax.set_xlabel("Sex"); ax.set_ylabel("Avg Charges")
                st.caption("Regional variations exist for smokers by gender.")
            elif question == "Age, BMI and Smoking Status Impact on Charges":
                sns.scatterplot(data=df, x='age', y='bmi', size='charges', hue='smoker',
                                palette={1:'red',0:'blue'}, alpha=0.6, ax=ax)
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='Yes', markerfacecolor='red', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='No', markerfacecolor='blue', markersize=6)
                ]
                ax.legend(title='Smoker', handles=legend_elements)
                st.caption("Smokers with high BMI and age show extreme costs.")
            elif question == "Obese Smokers vs Non-Obese Non-Smokers":
                dfCompare = df[(df['smoker_bmi'] >= 30) | (df['smoker_bmi']==0)]
                dfCompare['os_nons'] = np.where(dfCompare['smoker_bmi'] >=30,'Obese Smoker', 'Non-Obese Non-Smoker')
                sns.barplot(data=dfCompare, x='os_nons', y='charges', ci=None, palette='Set1', ax=ax)
                ax.set_xlabel("Group"); ax.set_ylabel("Average Charges")
                st.caption("Obese smokers are charged the most compared to others.")
            st.pyplot(fig)

    # --- Outliers ---
    elif analysis_type == "Outliers":
        with st.expander("🚨 Detect Outliers", expanded=True):
            question = st.selectbox("Select Question", [
                "Outlier Detection in Charges",
                "Outlier Detection in BMI"
            ])
            fig, ax = plt.subplots()
            if question == "Outlier Detection in Charges":
                sns.boxplot(x=df['charges'], color='lightblue', ax=ax)
                ax.set_xlabel("Charges")
                st.caption("Several outliers exist in charges, especially for smokers.")
            elif question == "Outlier Detection in BMI":
                sns.boxplot(x='bmi', data=df, color='lightgreen', ax=ax)
                ax.set_xlabel("BMI")
                st.caption("BMI shows moderate outliers around obesity range.")
            st.pyplot(fig)

    # --- Correlation ---
    elif analysis_type == "Correlation":
        with st.expander("📉 Correlation Analysis", expanded=True):
            question = st.selectbox("Select Question", [
                "Correlation Heatmap",
                "Correlation of Features with Charges"
            ])
            fig, ax = plt.subplots()
            if question == "Correlation Heatmap":
                numeric_cols = ['age','bmi','children','charges']
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.caption("Charges are moderately correlated with age, BMI and smoking.")
            elif question == "Correlation of Features with Charges":
                corr_with_charges = df.corr(numeric_only=True)['charges'].sort_values(ascending=False)
                sns.barplot(x=corr_with_charges.index, y=corr_with_charges.values, palette='viridis', ax=ax)
                ax.set_ylabel("Correlation coefficient")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                st.caption("Smoking status and BMI are the strongest predictors of charges.")
            st.pyplot(fig)


# --- Page 3: Predictions ---
elif page == "Predictions":
    st.title("Make a Prediction")

    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict"):
        # Transform inputs to match model features
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [1 if sex == "male" else 0],
            "bmi": [bmi],
            "children": [children],
            "smoker": [1 if smoker == "yes" else 0],
            "region_northwest": [1 if region == "northwest" else 0],
            "region_southeast": [1 if region == "southeast" else 0],
            "region_southwest": [1 if region == "southwest" else 0],
            "bmi_category_normal": [1 if 18.5 <= bmi < 25 else 0],
            "bmi_category_overweight": [1 if 25 <= bmi < 30 else 0],
            "bmi_category_underweight": [1 if bmi < 18.5 else 0],
            "smoker_bmi": [bmi if smoker == "yes" else 0]
        })

        pred = model.predict(input_df)[0]
        st.markdown(
            f"<div class='result-box'>💰 Estimated Insurance Cost: ${pred:,.2f}</div>",
            unsafe_allow_html=True
        )

# --- Page 4: About Us ---
elif page == "About Us":
    st.title("About Us")
    st.markdown("""
    ## 👨‍💻 Team: Data Wizards  
    - 🧑‍🎓 **Abdullah Khatri** – Data Scientist in training  
    - 🚀 Passionate about data, ML, and building real-world projects
    - 🎓 From this project [Medical Insurance Cost Prediction], I worked with new concept of ML Flow Tracking, used config.toml to theme this pages.   
    """)
