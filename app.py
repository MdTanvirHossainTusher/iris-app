import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


# Load model prediction function
def make_prediction(data):
    # Scaling the input data
    scale = StandardScaler()
    X_test = scale.fit_transform(data)

    # Load model
    with open("iris.pkl", "rb") as f:
        model = pickle.load(f)

    # Make prediction
    y_test_pred = model.predict(X_test)
    return y_test_pred


# Create Streamlit app
def main():
    st.title("Iris Flower Prediction")

    # Input fields for sepal length, width, petal length, and width
    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    # Prepare input data
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_df = pd.DataFrame(
        input_data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )

    # Make prediction when 'Predict' button is clicked
    if st.button("Predict"):
        prediction = make_prediction(input_df)
        st.write(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
