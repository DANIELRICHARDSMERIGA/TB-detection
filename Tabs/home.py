"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st

def app():
    """This function create the home page"""
    
    # Add title to the home page
    st.title("Tuberculosis Type and Level Detector")

    # Add image to the home page
    st.image("./images/home.png")

    # Add brief describtion of your web app
    st.markdown(
    """<p style="font-size:20px;">
            Tuberculosis (TB) detection involves identifying the type of TB a person may have, which typically falls into two categories: latent TB infection and active TB disease. Latent TB infection means the bacteria are present in the body but are inactive, causing no symptoms and not being contagious. Active TB disease, on the other hand, indicates that the bacteria are multiplying in the body, leading to symptoms such as coughing, fever, weight loss, and fatigue. Diagnosis often includes a combination of tests like skin or blood tests, chest X-rays, and sputum tests to determine whether TB is present and, if so, whether it is latent or active. Treatment varies based on the type detected, with latent TB often requiring medication to prevent it from becoming active, while active TB usually necessitates a more intensive treatment regimen to cure the disease, with an accuracy of up to 98%.
        </p>
    """, unsafe_allow_html=True)