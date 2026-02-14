import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Classifier")
st.title("Heart Disease Prediction App")

# Load preprocessing objects
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# Model paths
model_files = {
    "Logistic Regression": "models/Logistic Regression.pkl",
    "Decision Tree": "models/Decision Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive Bayes.pkl",
    "Random Forest": "models/Random Forest.pkl",
    "XGBoost": "models/XGBoost.pkl"
}

# Load metrics
results_df = pd.read_csv("models/model_results.csv")

# Sidebar model selection
model_name = st.sidebar.selectbox("Choose Model", list(model_files.keys()))
model = joblib.load(model_files[model_name])

st.subheader("Model Metrics")
st.dataframe(results_df[results_df["Model"] == model_name])

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.write(df.head())

    # If target exists
    if "HeartDisease" in df.columns:
        y_true = df["HeartDisease"].map({"Yes":1,"No":0})
        df = df.drop("HeartDisease", axis=1)
    else:
        y_true = None

    # Apply encoders to categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale for LR & KNN
    if model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(df)
    else:
        X = df

    # Predict
    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds)

    # If target available → show confusion matrix
    if y_true is not None:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())