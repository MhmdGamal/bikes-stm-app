import streamlit as st
import pandas as pd 
import numpy as np 
from dummies import *
import joblib

model=joblib.load('model.h5')
scaler=joblib.load('scaler.h5')

st.title('bikes renting app')
st.info(' tryin to build a model on the bikes dataset')

#[temp,humidity,hour,is_rush_hour,month]+[season]+[weather]+[weekday]+[pod]

#[,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,]
#['temp', 'humidity', 'month', 'hour', 'season_Spring', 'season_Summer',
 #      'season_Winter', 'weather_Mist', 'weather_Rainy', 'weather_Snowy',
  #     'week_day_Monday', 'week_day_Saturday', 'week_day_Sunday',
  #     'week_day_Thursday', 'week_day_Tuesday', 'week_day_Wednesday',
  #     'period_evening', 'period_morning', 'period_night']

temp=st.number_input('Enter Temperature: ')
humidity=st.number_input('Enter humidity: ')
hour = st.slider('hour? ',0,24,15)
month =st.slider('month? ',1,12,7)
season_selection=st.selectbox('Season? ',['winter','spring','summer','fall'])
season=season_dummies[season_selection]
weather_selection = st.selectbox('weather? ',['clear','mist','rainy','snowy'])
weather=weather_dummies[weather_selection]
weekday_selection = st.selectbox('weekday? ',['saturday','sunday','monday','tuesday','wednesday','thursday','friday'])
weekday=weekdays_dummies[weekday_selection]
pod_selection=st.selectbox('pod?',['evening','morning','night','afternoon'])
pod=pod_dummies[pod_selection]

data=[temp,humidity,month,hour]+season+weather+weekday+pod

data_scaled=scaler.transform([data])

result=model.predict(data_scaled)

st.write(result)


