import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset and model once
@st.cache_data
def load_data():
    return pd.read_csv('data/winequality-red.csv')

@st.cache_data
def load_test_data():
    return pd.read_csv('data/test_data.csv')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🍷 Wine Quality Prediction App")
    st.write("""
    This app predicts whether a wine is of Good quality based on its chemical properties.
    Explore the data, visualise feature relationships, and use the trained ML model to predict wine quality.
    """)

    menu = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
    )

    data = load_data()
    model = load_model()

    if menu == "Home":
        st.header("Welcome!")
        st.write("Use the sidebar to explore different sections of the app.")

    elif menu == "Data Exploration":
        st.header("Data Exploration")
        st.write("Dataset shape:", data.shape)
        st.write("Columns and data types:")
        st.write(data.dtypes)
        st.write("Sample data:")
        st.dataframe(data.sample(10))
        
        st.subheader("Filter Data")
        alcohol_range = st.slider("Alcohol %", float(data['alcohol'].min()), float(data['alcohol'].max()), (float(data['alcohol'].min()), float(data['alcohol'].max())))
        filtered_data = data[(data['alcohol'] >= alcohol_range[0]) & (data['alcohol'] <= alcohol_range[1])]
        st.write(f"Filtered data shape: {filtered_data.shape}")
        st.dataframe(filtered_data.head(10))

    elif menu == "Visualisations":
        st.header("Visualisations")
        st.subheader("Histogram of Wine Quality")
        fig1 = px.histogram(data, x='quality', nbins=10, color='quality')
        st.plotly_chart(fig1)

        st.subheader("Correlation Heatmap")
        corr = data.corr()
        fig2, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', ax=ax)
        st.pyplot(fig2)

        st.subheader("Alcohol vs Quality Boxplot")
        fig3, ax = plt.subplots()
        sns.boxplot(x='quality', y='alcohol', data=data, ax=ax)
        st.pyplot(fig3)

    elif menu == "Model Prediction":
        st.header("Predict Wine Quality")

        st.write("Enter chemical properties of the wine:")

        # Input widgets for features
        fixed_acidity = st.number_input("Fixed Acidity", float(data['fixed acidity'].min()), float(data['fixed acidity'].max()), float(data['fixed acidity'].mean()))
        volatile_acidity = st.number_input("Volatile Acidity", float(data['volatile acidity'].min()), float(data['volatile acidity'].max()), float(data['volatile acidity'].mean()))
        citric_acid = st.number_input("Citric Acid", float(data['citric acid'].min()), float(data['citric acid'].max()), float(data['citric acid'].mean()))
        residual_sugar = st.number_input("Residual Sugar", float(data['residual sugar'].min()), float(data['residual sugar'].max()), float(data['residual sugar'].mean()))
        chlorides = st.number_input("Chlorides", float(data['chlorides'].min()), float(data['chlorides'].max()), float(data['chlorides'].mean()))
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", float(data['free sulfur dioxide'].min()), float(data['free sulfur dioxide'].max()), float(data['free sulfur dioxide'].mean()))
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", float(data['total sulfur dioxide'].min()), float(data['total sulfur dioxide'].max()), float(data['total sulfur dioxide'].mean()))
        density = st.number_input("Density", float(data['density'].min()), float(data['density'].max()), float(data['density'].mean()))
        pH = st.number_input("pH", float(data['pH'].min()), float(data['pH'].max()), float(data['pH'].mean()))
        sulphates = st.number_input("Sulphates", float(data['sulphates'].min()), float(data['sulphates'].max()), float(data['sulphates'].mean()))
        alcohol = st.number_input("Alcohol", float(data['alcohol'].min()), float(data['alcohol'].max()), float(data['alcohol'].mean()))

        if st.button("Predict Quality"):
            input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                                        total_sulfur_dioxide, density, pH, sulphates, alcohol]])
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)

            quality_label = "Good Quality Wine 🍷" if prediction[0] == 1 else "Bad Quality Wine"
            confidence = prediction_proba[0][prediction[0]] * 100

            st.success(f"Prediction: {quality_label}")
            st.info(f"Confidence: {confidence:.2f}%")

    elif menu == "Model Performance":
        st.header("Model Performance")

        # Load test data with labels
        test_data = load_test_data()
        X_test = test_data.drop('quality_label', axis=1)
        Y_test = test_data['quality_label']

        # Predict on test set
        predictions = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(Y_test, predictions)
        st.write(f"Accuracy: {acc:.2f}")

        st.subheader("Classification Report")
        report = classification_report(Y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(Y_test, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
