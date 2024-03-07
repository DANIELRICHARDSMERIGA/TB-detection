"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Prediction of Stress Level.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    Rpm = st.slider("Respiration per minute", int(df["Respiration_per_minute"].min()), int(df["Respiration_per_minute"].max()))
    Age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))
    Gender = st.slider("Gender", int(df["Gender"].min()), int(df["Gender"].max()))
    Weight = st.slider("Weight", float(df["Weight"].min()), float(df["Weight"].max()))
    BMI = st.slider("Body Mass Index", float(df["Body_Mass_Index"].min()), float(df["Body_Mass_Index"].max()))
    Cough = st.slider("Cough", int(df["Cough"].min()), int(df["Cough"].max()))
    Dyspnea = st.slider("Dyspnea", int(df["Dyspnea"].min()), int(df["Dyspnea"].max()))
    Fever = st.slider("Fever", int(df["Fever"].min()), int(df["Fever"].max()))
    Weight_Loss = st.slider("Weight_Loss", int(df["Weight_Loss"].min()), int(df["Weight_Loss"].max()))
    Chest_Pain = st.slider("Chest_Pain", int(df["Chest_Pain"].min()), int(df["Chest_Pain"].max()))
    Hemoptysis = st.slider("Hemoptysis", int(df["Hemoptysis"].min()), int(df["Hemoptysis"].max()))
    Exposure_to_TB = st.slider("Exposure_to_TB", int(df["Exposure_to_TB"].min()), int(df["Exposure_to_TB"].max()))
    Diabetes = st.slider("Diabetes", int(df["Diabetes"].min()), int(df["Diabetes"].max()))
    HIV_Status = st.slider("HIV_Status", int(df["HIV_Status"].min()), int(df["HIV_Status"].max()))
    Smoking_History = st.slider("Smoking_History", int(df["Smoking_History"].min()), int(df["Smoking_History"].max()))
    Alcohol_Use = st.slider("Alcohol_Use", int(df["Alcohol_Use"].min()), int(df["Alcohol_Use"].max()))

    # Create a list to store all the features
    features = [Rpm,Age,Gender,Weight,BMI,Cough,Dyspnea,Fever,Weight_Loss,Chest_Pain,Hemoptysis,Exposure_to_TB,Diabetes,HIV_Status,Smoking_History,Alcohol_Use]


    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        

        # Print the output according to the prediction
        if (prediction == 0):
            st.warning("The person has risk of extrapulmonary TB")
            st.info("Smell some Eucalyptus oil")
        elif (prediction == 1):
            st.warning("The person has risk of Pulmonary TB")
            st.info("Requires medical attention and nebulizaton")
        elif (prediction == 2):
            st.error("The person has high risk of Chronic / Drug Resistant Tuberculosis")
            st.info("Admit to ICU and start ventilation")
        elif (prediction == 3):
            st.error("The person has pulomonary Fungucitis!")
            st.info("Require Amycline or similar antibiotics")
        else:
            st.success("The person has no lungs problems 😄")

        # Print teh score of the model 
        st.write("The model used is trusted by doctor and has an accuracy of ", (score*100),"%")
