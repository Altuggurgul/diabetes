import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Simple Diabetes Prediction App
""")

st.sidebar.header('User Input Parameters')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.read_csv(uploaded_file,skipinitialspace=True,usecols=columns)
else:
    def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies', 0.1, 20.0, 2.0)
        Glucose = st.sidebar.slider('Glucose', 1.0, 200.0, 117.0)
        BloodPressure = st.sidebar.slider('BloodPressure', 10.0, 150.0, 60.0)
        SkinThickness = st.sidebar.slider('SkinThickness', 0.1, 99.0, 40.0)
        Insulin = st.sidebar.slider('Insulin', 2.0, 846.0, 100.0)
        BMI = st.sidebar.slider('BMI', 1.0, 35.0, 70.0)
        DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.1, 998.0, 200.0)
        Age = st.sidebar.slider('Age', 10.0, 80.0, 35.0)
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
sheet_url = 'https://docs.google.com/spreadsheets/d/1OMurqWKO_E6IJHzX9HfvMssccJWpCspnA_B0KgLguPM/edit#gid=17834267'
csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
diabetes_raw = pd.read_csv(csv_export_url)
diabetes = diabetes_raw.drop(columns=['Outcome'])
df = pd.concat([input_df,diabetes],axis=0)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('diabetes_simple.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['N-Diabetes','Diabetes'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


if st.button('Feature Importances'):

    sheet_url = 'https://docs.google.com/spreadsheets/d/1OMurqWKO_E6IJHzX9HfvMssccJWpCspnA_B0KgLguPM/edit#gid=17834267'
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)

    image = Image.open("importances.png")
    st.image(image, use_column_width=True)

    # st.write("Dataframe Head")
    # st.write(df.head())
    # st.write("Shape")
    #
    # st.write(df.shape)
    # st.write("Describe")
    # st.write(df.describe().T)
    #
    # st.write("Histogram")
    #
    # with sns.axes_style("white"):
    #     df.hist(bins=10, figsize=(10, 10))
    #
    # st.pyplot()
    #
    # st.write("Intercorrelation Matrix Heatmap")
    #
    # corr = df.corr()
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True
    # with sns.axes_style("white"):
    #     f, ax = plt.subplots(figsize=[18, 13])
    #     ax = sns.heatmap(corr,mask=mask, vmax=1, square=True, cmap="magma",annot=True)
    # st.pyplot()