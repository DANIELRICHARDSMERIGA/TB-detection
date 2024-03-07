import streamlit as st
import pandas as pd
import numpy as np
import random
import imagerec
import streamlit.components.v1 as components
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
import time
import tensorflow
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

def app():
    components.html(
        """
        <style>
            #effect{
                margin:0px;
                padding:0px;
                font-family: "Source Sans Pro", sans-serif;
                font-size: max(8vw, 20px);
                font-weight: 700;
                top: 0px;
                right: 25%;
                position: fixed;
                background: -webkit-linear-gradient(0.25turn,#FF4C4B, #FFFB80);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p{
                font-size: 2rem;
            }
        </style>
        <p id="effect">Tuber-Detector</p>
        """,
        height=69,
    )


    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)

    st.title("Tuberculosis Detector")

    st.write('<style>div.row-widget.stMarkdown { font-size: 1.2rem; }</style>', unsafe_allow_html=True)



    uploaded_file = st.file_uploader("Choose a File", type=['jpg','png','jpeg'])


    if uploaded_file!=None:
        st.image(uploaded_file)
    else:
        st.info("Please upload an image to test")
    x = st.sidebar.button("Detect Tuberculosis")
    if uploaded_file is not None:    
        image1 = Image.open(uploaded_file)

    img_height , img_width = 150,150
    if x:
        if uploaded_file is None:
            st.write("Invalid command, please upload an image")
        else:
            img = load_img(uploaded_file,target_size=(img_height,img_width))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array,axis=0)
            img_array /= 255.
            classifier_model = "Tuberculosis_new.h5"
            model = load_model(classifier_model)
            prediction = model.predict(img_array)
            score = np.load('model_acc_new.npy')
            if prediction[0][0] > 0.5:
                st.warning("Tuberculosis Detected")
                st.info(str(score*100)+ "% Confidence Level")
            else:
                st.warning("Normal Case")
                st.info(str(score*100)+ "% Confidence Level")
    
            

