import numpy as np 
import streamlit as st 
import joblib 
from sklearn.datasets import load_iris 
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸", layout="centered") 
st.title("🌸 Iris Flower Prediction App") 
st.write("Enter flower measurements and predict the Iris type using a trained KNN model.") 
# Load iris labels (for names only) 
iris = load_iris() 
# Load saved model and scaler 
model = joblib.load("knn_iris_model.pkl") 
scaler = joblib.load("scaler.pkl") 
st.subheader(" Input Features (in cm)") 
col1, col2 = st.columns(2) 
with col1: 
    sepal_length = st.number_input("Sepal Length", min_value=0.0, value=5.1, step=0.1) 
    sepal_width = st.number_input("Sepal Width", min_value=0.0, value=3.5, step=0.1) 
with col2: 
    petal_length = st.number_input("Petal Length", min_value=0.0, value=1.4, step=0.1) 
    petal_width = st.number_input("Petal Width", min_value=0.0, value=0.2, step=0.1) 
if st.button("Predict 🌼"): 
    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) 
    user_scaled = scaler.transform(user_data) 
    pred = model.predict(user_scaled)[0] 
    confidence = model.predict_proba(user_scaled).max() 
    st.success(f"Predicted Flower: **{iris.target_names[pred].title()}**") 
    st.info(f"Confidence: **{confidence*100:.2f}%**") 
st.markdown("---") 
st.caption("Built with Streamlit • KNN • Iris Dataset") 
