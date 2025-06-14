import streamlit as st #streamlit库
import joblib #模型持久化库
import numpy as np #数值计算库
import pandas as pd #数据处理和分析库
import shap #模型可解释性库
import matplotlib.pyplot as plt #数据可视化库
import sklearn #机器学习库
import lightgbm #算法库
import statsmodels #统计建模库

# Load the model
model = joblib.load('LightGBM_label.pkl')

# Define feature options
Volume_stage_options = {
    1: '1 stage (1)',
    2: '2 stage (2)',
    3: '3 stage (3)'
}

# Define feature names
feature_names = ["Volume_stage","ISL_stage","Hyperlipidemia","Diabetes",
                 "Disease_duration","Total_lower_limb_volume","TBW_FFM","BIA_50kHz","Percentage_of_body_fat"
]

# Streamlit user interface
st.title("Lymphoscintigraphy Stage Predictor")

# Volume_stage: categorical selection
Volume_stage = st.selectbox("Chest pain type:", options=list(Volume_stage_options.keys()), format_func=lambda x: Volume_stage_options[x])
# ISL_stage: categorical selection
ISL_stage = st.selectbox("ISL_stage (0=1-2a stage, 1=2b-3 stage):", options=[0, 1], format_func=lambda x: '1-2a stage (0)' if x == 0 else '2b-3 stage (1)')
# Hyperlipidemia: categorical selection
Hyperlipidemia = st.selectbox("Hyperlipidemia:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# Diabetes: categorical selection
Diabetes = st.selectbox("Hyperlipidemia:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# Disease_duration: numerical input
Disease_duration = st.number_input("Disease_duration:", min_value=1, max_value=360, value=120)# 设置默认值为120个月
# Total_lower_limb_volume: numerical input
Total_lower_limb_volume = st.number_input("Total_lower_limb_volume:", min_value=1, max_value=200, value=120)
# TTBW_FFM: numerical input
TBW_FFM = st.number_input("TBW_FFM:", min_value=70, max_value=80, value=75)
# BIA_50kHz: numerical input
BIA_50kHz = st.number_input("BIA_50kHz:", min_value=0.5, max_value=5, value=3)
# Percentage_of_body_fat: numerical input
Percentage_of_body_fat = st.number_input("Percentage_of_body_fat:", min_value=15, max_value=60, value=20)

# Process inputs and make predictions
feature_values = [Volume_stage, ISL_stage, Hyperlipidemia, Diabetes, Disease_duration, Total_lower_limb_volume,TBW_FFM, BIA_50kHz,Percentage_of_body_fat]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
if predicted_class == 1:
    advice = (
        f"According to our model, you have a high risk of advanced lymphoscintigraphy stage. "
        f"The model predicts that your probability of having advanced lymphoscintigraphy stage is {probability:.1f}%. "
        "While this is just an estimate, it suggests that you may be at significant risk. "
        )
    else:
    advice = (
        f"According to our model, you have a low risk of advanced lymphoscintigraphy stage. "
        f"The model predicts that your probability of not having advanced lymphoscintigraphy stage is {probability:.1f}%. "
        "However, maintaining a healthy lifestyle is still very important. "
        )
st.write(advice)
# Calculate SHAP values and display force plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_force_plot.png")