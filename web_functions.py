"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('tuberculosis_dataset.csv')

    #data preprocessing
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    print(df["HIV_Status"].unique())
    df['HIV_Status'] = LabelEncoder().fit_transform(df['HIV_Status'])
    print(df["Smoking_History"].unique())
    df['Smoking_History'] = LabelEncoder().fit_transform(df['Smoking_History'])
    #df['TB_Diagnosis'] = LabelEncoder().fit_transform(df['TB_Diagnosis'])

    mapping={'Risk of Extrapulmonary TB':0,
    'Risk of Pulmonary TB':1,
    'High Risk of Chronic/Drug Resistant Tuberculosis':2,
    'Pulmonary Fungucitis':3,
    'No Lung Problems':4}

    df["TB_Diagnosis"] = df['TB_Diagnosis'].map(mapping)

    # Perform feature and target split
    X = df[["Respiration_per_minute","Age","Gender","Weight","Body_Mass_Index","Cough","Dyspnea","Fever","Weight_Loss","Chest_Pain","Hemoptysis","Exposure_to_TB","Diabetes","HIV_Status","Smoking_History","Alcohol_Use"]]
    y = df['TB_Diagnosis']

    return df, X, y

@st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    model = RandomForestClassifier(n_estimators=100)

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # Fit the data on model
    model.fit(x_train, y_train)
    # Get the model score
    score = model.score(x_train, y_train)

    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
