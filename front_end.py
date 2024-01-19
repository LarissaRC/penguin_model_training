# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained KNN model
loaded_model_knn = joblib.load('knn_model.pkl')

# Load the dataset
df = pd.read_csv('penguins_size.csv')
df = df.dropna()

# Streamlit app
def main():
    st.title("KNN Classifier and Dataset Visualization")

    # Display the loaded dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Sidebar for model testing
    st.sidebar.subheader("Test the Model")
    culmen_length = st.sidebar.slider("Culmen Length (mm)", df['culmen_length_mm'].min(), df['culmen_length_mm'].max())
    culmen_depth = st.sidebar.slider("Culmen Depth (mm)", df['culmen_depth_mm'].min(), df['culmen_depth_mm'].max())
    flipper_length = st.sidebar.slider("Flipper Length (mm)", df['flipper_length_mm'].min(), df['flipper_length_mm'].max())

    # Predict using the loaded KNN model
    test_data = [[culmen_length, culmen_depth, flipper_length]]
    prediction = loaded_model_knn.predict(test_data)

    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"The predicted class is: {prediction[0]}")

    # Display the dataset visualization
    st.subheader("Dataset Visualization")

    # Plot the relation of each feature with each class
    features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']
    classes = df['species'].unique()

    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(feature)
        ax.set_ylabel('Class')

        for class_label in classes:
            class_data = df[df['species'] == class_label]
            ax.scatter(class_data[feature], class_data['species'], label=class_label)

        ax.legend(loc='best', prop={'size': 8})
        st.pyplot(fig)

if __name__ == '__main__':
    main()
