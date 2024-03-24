import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


def make_prediction(data):

    scale = StandardScaler()
    X_test = scale.fit_transform(data)

    # Load model
    with open("model/iris_rf.pkl", "rb") as f:
        model = pickle.load(f)

    y_test_pred = model.predict(X_test)
    return y_test_pred


def main():
    st.title("Iris Flower Prediction")

    # Take input
    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_df = pd.DataFrame(
        input_data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    classes = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    if st.button("Predict"):
        prediction = make_prediction(input_df)
        st.write(f"Predicted class: {classes[prediction[0]]}")


if __name__ == "__main__":
    main()
