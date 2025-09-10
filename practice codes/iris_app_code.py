import streamlit as st
import numpy as np
import pandas as pd
import joblib 

st.title("My model app")
st.write("this site gives prediction to Iris species")
st.header("Please input your features")

st.sidebar.header("Input Features")
st.feature1 = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
st.feature2 = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
st.feature3 = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
st.feature4 = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

st.write(st.feature1, st.feature2, st.feature3, st.feature4)
st.write(st.feature1 + st.feature2 + st.feature3 + st.feature4)

model = joblib.load(r"C:\Users\Admin\Documents\Datascience\practice codes\model")

dict_data = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}
model_input = np.array([[st.feature1, st.feature2, st.feature3, st.feature4]])

model_input = model_input.reshape(1, -1)
prediction = model.predict(model_input)

st.write("### Prediction")
st.write(f"The predicted specie is: **{dict_data[prediction[0]]}**")