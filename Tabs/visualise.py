"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
'''from sklearn.metrics import plot_confusion_matrix'''
from sklearn import tree
import streamlit as st


# Import necessary functions from web_functions
from web_functions import train_model
def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Stress Level")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize = (15, 10))
        ax = sns.heatmap(df.iloc[:, :].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)

    # if st.checkbox("Show Scatter Plot"):
        
    #     figure, axis = plt.subplots(2, 2,figsize=(15,10))

    #     sns.scatterplot(ax=axis[0,0],data=df,x='AGE',y='copd')
    #     axis[0, 0].set_title("Breathing complexity with respect to age")
  
    #     sns.scatterplot(ax=axis[0,1],data=df,x='MWT1',y='FEV1PRED')
    #     axis[0, 1].set_title("Mean Whooping time vs Fibrosis Prediction")
  
    #     sns.scatterplot(ax=axis[1, 0],data=df,x='SGRQ',y='HAD')
    #     axis[1, 0].set_title("SGRQ vs HAD")
  
    #     sns.scatterplot(ax=axis[1,1],data=df,x='Resp_pm',y='AGE')
    #     axis[1, 1].set_title("Respiration Per Minute vs Patient Age")
    #     st.pyplot()

    if st.checkbox("Display Boxplot"):
        fig, ax = plt.subplots(figsize=(15,5))
        df.boxplot(["Respiration_per_minute","Age","Gender","Weight","Body_Mass_Index","Cough","Dyspnea","Fever","Weight_Loss","Chest_Pain","Hemoptysis","Exposure_to_TB","Diabetes","HIV_Status","Smoking_History","Alcohol_Use"],ax=ax)
        st.pyplot()

    if st.checkbox("Show Sample Results"):
        safe = (df['TB_Diagnosis'] == 0).sum()
        low = (df['TB_Diagnosis'] == 1).sum()
        med = (df['TB_Diagnosis'] == 2).sum()
        high = (df['TB_Diagnosis'] == 3).sum()
        vhigh = (df['TB_Diagnosis'] == 4).sum()
        data = [safe,low]
        data = [safe, low]
        labels = ['yes','no']
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        st.pyplot()

    

    
    
