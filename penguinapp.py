import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

st.write('This is the day that you will always')

st.sidebar.header('Input fast or die')
st.sidebar.markdown('Alfabetafla')

uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider("Bill length", 32.1, 59.6, 43.9)
        bill_depth = st.sidebar.slider('Bill depth', 13.1,21.5, 17.2)
        flipper_length = st.sidebar.slider('Flipper length', 172.0, 231.0, 201.0)
        body_mass = st.sidebar.slider('Body mass', 2700.0, 6300.0, 4207.0)
       
        data = {'island':island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm' : bill_depth,
        'flipper_length_mm': flipper_length,
        'body_mass_g':body_mass
         }
        features = pd.DataFrame(data, index = [0])
        return features
    input_df = user_input_features()


penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop('species', axis = 1)
df = pd.concat([input_df, penguins], axis = 0)
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df= pd.concat([df, dummy], axis = 1)
    del df[col]
df=df[:1]





st.subheader('User INput featuring vs')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('You can use this slidebars')
    st.write(df)

load_clf = pickle.load(open('modeling.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('PRedestionation')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
