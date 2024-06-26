import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title("Classifying Iris Flowers")
st.markdown("Toy Moddel to play to classify iris (setosa, versicolor,vrginica)based on sepal and petal length and width ")
st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal Length(cm), 1.0, 8.0, 0.5')
    sepal_w = st.slider('Sepal Width(cm), 2.0,4.4,0.5')

with col2:
    st.text("Petal Characteristics")
    petal_l = st.slider('Petal Length(cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal Width(cm)', 0.1, 2.5, 0.5)

st.text("")
if st.button('Predict type of Iris'):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])

st.text('')
st.text('')

st.markdown(
    '''
    Created by Muskan Shrestha @2024  
    [Github: https://github.com/MuskanShrestha58](https://github.com/MuskanShrestha58)
    '''
)
