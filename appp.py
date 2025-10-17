# all_in_one_medical_diagnostics.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# ===========================
# 1. Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="AI HealthMate",
    page_icon="❤️",
    layout="wide"
)

st.title("🩺 AI HealthMate – All-in-One Medical Diagnostics")
st.markdown("Fast & smart medical assistant using ML + Transformers")

# ===========================
# 2. Load Models (Cached)
# ===========================
@st.cache_resource
def load_summarizer_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_summarizer_model()

@st.cache_resource
def load_risk_model():
    # Simple example: RandomForest trained on synthetic heart/diabetes data
    # Replace with real dataset
    X = np.random.rand(100,5)
    y = np.random.randint(0,2,100)
    clf = RandomForestClassifier()
    clf.fit(X,y)
    return clf

risk_model = load_risk_model()

# ===========================
# 3. Sidebar Navigation
# ===========================
option = st.sidebar.selectbox(
    "Choose Diagnostic Module:",
    ["Symptom Analyzer", "Medical Report Summarizer", "Health Risk Predictor", "Health Advisor Chat"]
)

# ===========================
# 4. Symptom Analyzer
# ===========================
if option == "Symptom Analyzer":
    st.header("📝 Symptom-Based Disease Prediction")
    symptoms = st.text_area("Enter symptoms separated by commas", "")
    if st.button("Predict Disease"):
        if symptoms.strip() == "":
            st.warning("Please enter symptoms!")
        else:
            # Example ML prediction (replace with real embeddings + classifier)
            possible_diseases = {
                "Common Cold": np.random.randint(50,90),
                "Flu": np.random.randint(30,80),
                "COVID-19": np.random.randint(10,60)
            }
            st.subheader("Possible Diseases:")
            for disease, prob in possible_diseases.items():
                st.write(f"{disease}: {prob}%")

# ===========================
# 5. Medical Report Summarizer
# ===========================
elif option == "Medical Report Summarizer":
    st.header("📄 Medical Report Summarization")
    report = st.text_area("Paste medical report text here", height=200)
    
    def summarize_text(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    if st.button("Summarize Report"):
        if report.strip() == "":
            st.warning("Please enter report text!")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_text(report)
            st.subheader("Summary:")
            st.write(summary)

# ===========================
# 6. Health Risk Predictor
# ===========================
elif option == "Health Risk Predictor":
    st.header("❤️ Health Risk Prediction")
    age = st.number_input("Age", 1, 120, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    bp = st.number_input("Blood Pressure", 60, 200, 120)
    glucose = st.number_input("Glucose Level", 50, 250, 90)
    sleep = st.number_input("Hours of Sleep per Day", 0, 24, 7)
    
    if st.button("Predict Risk"):
        features = np.array([[age, bmi, bp, glucose, sleep]])
        risk = risk_model.predict_proba(features)[0][1]
        st.subheader("Predicted Health Risk:")
        st.write(f"{risk*100:.1f}%")
        # Visualize with bar
        st.progress(int(risk*100))

# ===========================
# 7. Health Advisor Chat
# ===========================
elif option == "Health Advisor Chat":
    st.header("💬 Health Advice Chat")
    user_query = st.text_input("Ask a general health question (no medical diagnosis)")
    
    @st.cache_resource
    def load_advice_pipeline():
        return pipeline("text-generation", model="distilgpt2")
    
    advice_model = load_advice_pipeline()
    
    if st.button("Get Advice"):
        if user_query.strip() == "":
            st.warning("Please ask a question!")
        else:
            response = advice_model(user_query, max_length=100, do_sample=True, temperature=0.7)
            st.subheader("Advice:")
            st.write(response[0]["generated_text"])
