import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("autoscout.pkl", "rb"))
enc = pickle.load(open("autoscout_encoder.pkl", "rb")) # 
df = pd.read_csv("final_model.csv")


st.markdown("# <center>AutoScout Car Price Predictor</center>", unsafe_allow_html=True )
st.markdown("#### <center>This app predicts the **car prices**!</center>", unsafe_allow_html=True )

st.sidebar.header('User Input Features')

# Collects user input features into dataframe

user_make_model = st.sidebar.selectbox("Select your car's Make&Model", df.make_model.unique())
user_body_type = st.sidebar.selectbox("Select your car's Body Type", df.body_type.unique())

user_gear = st.sidebar.selectbox("Select your car's Gearing Type", df["Gearing Type"].unique())
user_fuel = st.sidebar.selectbox("Select your car's Fuel Type", df.Fuel.unique())

user_km = st.sidebar.slider("KM", 0, 300000, step=10000)
user_age = int(st.sidebar.selectbox("Age", (0,1,2,3)))

user_cc = st.sidebar.number_input("Displacement (cc)", 900, 2967, 1200, 100)

user_hp = st.sidebar.number_input("HP", 55, 390, 90, 10)

car = pd.DataFrame({"make_model" : [user_make_model],
                    "body_type" : [user_body_type],
                    "km" : [user_km],
                    "hp" : [user_hp],
                    "Gearing Type" : [user_gear],
                    "Displacement_cc" : [user_cc],
                    "Fuel": [user_fuel],
                    "Age" : [user_age]})


# we created a df with the data taken from the users

cat = car.select_dtypes("object").columns
car[cat] = enc.transform(car[cat])

st.subheader('Prediction')

c1, c2, c3, c4, c5,c6,c7,c8,c9 = st.columns(9) 
if c1.button('Predict'):
    result = model.predict(car)[0]
    st.info(f"Predicted value of your car : ${round(result)}")



