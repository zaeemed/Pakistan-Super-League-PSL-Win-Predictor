import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
# import Image from pillow to open images
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import numpy as np
import pandas as pd

# Unpickling the trained model
xgbc_model = pickle.load(open("./PSL-Win-XGBC-model.pkl", "rb"))

# Title
st.markdown("<h1 style = 'color:Gold; Text-align: Center; font-size: 40px;'>Pakistan Super League (PSL) Win Predictor</h1>", unsafe_allow_html=True)

img = Image.open("./PSL-6.jpg")
  
# display image using streamlit
# width is used to set the width of an image
st.image(img, width = 700)

form = st.sidebar.form(key='my_form')
# Add a selectbox to the sidebar:
Team1 = form.selectbox(
    'Select Team Batting First',
    ('Islamabad United', 'Karachi Kings', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
)

Team2 = form.selectbox(
    'Select Team Batting Second',
    ('Karachi Kings', 'Islamabad United', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
)


## Target for the team batting 2nd
target = form.text_input('Target For the Team Batting Second', 110)

## Current runs of the team batting 2nd
cur_runs = form.text_input('Current Runs of the Team Batting Second', 10)

## Current wickets of the team batting 2nd
wickets = form.text_input('Current Wickets of the Team Batting Second', 5.0)

## Current overs of the team batting 2nd
overs = form.text_input('Current Overs Played by the Team Batting Second', 5.5)
submit_button = form.form_submit_button(label = 'Predict Win %')

if submit_button:
    target = float(target)
    cur_runs = float(cur_runs)
    wickets = float(wickets)
    overs = float(overs)

    input_data = {
        "wickets": wickets,
        "balls_left" : 120 - ((overs - overs%1) * 6 + (overs%1)*10),
        "runs_left":target - cur_runs
    }
    input_data_df = pd.DataFrame(input_data,index=[0])
    prediction = xgbc_model.predict_proba(input_data_df)
    # Create a pieplot
    print(prediction)
    fig = px.pie(prediction, values = prediction[0][:], names=[Team1, Team2], title='Match winning percentage for both the Teams')
    st.plotly_chart(fig)
    st.success("Interpretation : There is a "+str(round(prediction[0][0] * 100))+ "% chance the team batting second ("+ Team2 +") is going to lose (or the first team ("+ Team1 +") is going to win) and a " + str(round(prediction[0][1] * 100))+"% chance that ("+ Team2 +") will win.")





