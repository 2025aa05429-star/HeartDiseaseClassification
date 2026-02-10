import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Classifier")

st.title("❤️ Heart Disease Prediction App")

st.write("Upload test dataset and choose a model to predict heart disease.")

# Load saved scaler
scaler = joblib.load("models/scaler.pkl")

# Available models
model_files = {
    "Logistic Regression": "models/Logistic Regression.pkl",
    "Decision Tree": "models/Decision Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive Bayes.pkl",
    "Random Forest": "models/Random Forest.pkl",
    "XGBoost": "models/XGBoost.pkl"
}

# Load evaluation metrics table
results_df = pd.read_csv("models/model_results.csv")

# Sidebar model selection
st.sidebar.header("Select Model")
model_name = st.sidebar.selectbox("Choose a model", list(model_files.keys()))

model = joblib.load(model_files[model_name])

# Show model metrics
st.subheader("📊 Model Evaluation Metrics")
st.dataframe(results_df[results_df["Model"] == model_name])

# File uploader
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    # Check if target column present
    if "target" in data.columns:
        X = data.drop("target", axis=1)
        y_true = data["target"]
    else:
        X = data
        y_true = None

    # Scale for LR + KNN
    if model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X)

    st.subheader("Predictions")
    st.write(y_pred)

    # If target exists → show confusion matrix
    if y_true is not None:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())