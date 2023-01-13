import streamlit as st
import hopsworks
import joblib
import pandas as pd
import geopandas
import numpy as np
import folium
from folium import plugins
from streamlit_folium import st_folium, folium_static
import json
import time
from datetime import timedelta, datetime
from branca.element import Figure
import altair as alt
import os 
from functions.functions import decode_features, get_model
from functions.get_weather_data import get_weather_data

st.set_page_config(layout="wide")

def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True )


st.title('PM10 Predictions for Poland🇵🇱')

col1, col2 = st.columns(2)

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.sidebar.write(36 * "-")

st.sidebar.header('\n📡 Connecting to Hopsworks Feature Store...')

project = hopsworks.login(project = "ID2223_Ernest", api_key_value=os.environ['HOPSWORKS_API_KEY'])
fs = project.get_feature_store()
feature_view = fs.get_feature_view(
    name = 'poland_air_quality_fv',
    version = 1
)

st.sidebar.write("Successfully connected!✔️")
progress_bar.progress(20)

st.sidebar.write(36 * "-")
st.sidebar.header('\n☁️ Collecting the weather forecasts...')

# GET THE WEATHER DATA ----------------------------------------------------------------------------------------------------------------------------------------------------------

# check if it was previously downloaded
weather_fname = f"air_quality/weather_files/{str(datetime.now())[:10]}.csv"
if os.path.exists(weather_fname):
    weather_data = pd.read_csv(weather_fname)
# doesn't exist, download
else:
    weather_data = get_weather_data()

st.sidebar.write("Collected!✔️")
progress_bar.progress(60)

st.sidebar.write(36 * "-")
st.sidebar.header(f"🤖Loading the models...")

fig = Figure(width=1000,height=1000)
df = pd.read_excel("air_quality/data_poland/meta.xlsx")

geometry = geopandas.points_from_xy(df.lat, df.lon)
geo_df = geopandas.GeoDataFrame(
    df, geometry=geometry
)


# GET ACTUAL PREDICTIONS ----------------------------------------------------------------------------------------------------------------------------------------------------------

data_pred = weather_data[['date', 'city']].copy()

X = weather_data[['tempmax', 'tempmin', 'temp', 'feelslikemax',
       'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
       'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
       'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
       'conditions', 'area', 'density', 'no_people']]


model_gb = get_model(project=project,
                  model_name="Gradient_Duster",
                  evaluation_metric="mae",
                  sort_metrics_by="max")

model_rfr = get_model(project=project,
                  model_name="AirParticle_Forest",
                  evaluation_metric="mae",
                  sort_metrics_by="max")

model_lasso = get_model(project=project,
                  model_name="PM10Lasso",
                  evaluation_metric="mae",
                  sort_metrics_by="max")

st.sidebar.write("Models loaded!✔️")

data_pred['Gradient_Duster'] = [int(x) for x in model_gb.predict(X)]
data_pred['AirParticle_Forest'] = [int(x) for x in model_rfr.predict(X)]
data_pred['PM10Lasso'] =[int(x) for x in  model_lasso.predict(X)]
data_pred['Mean'] = data_pred[['Gradient_Duster', 'AirParticle_Forest', 'PM10Lasso']].mean(axis=1).apply(lambda x: int(x))

st.sidebar.write(36 * "-")
st.sidebar.header(f"🗺️Rendering the map...")

with col1:

    radio_options = ['Today', 'Tomorrow']
    for i in range(5):
        radio_options.append(str(datetime.now() + timedelta(days=2+i))[:10])
        
    forecast_day_option = st.radio('Select forecasting day',
                                    radio_options, horizontal=True)

m = folium.Map(location=[52.232222, 19.508333], tiles="Stamen Toner", zoom_start=6,
               zoom_control=False,
               scrollWheelZoom=False,
               dragging=False)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]


if forecast_day_option == "Today":
    forecast_day_option = str(datetime.now())[:10]
elif forecast_day_option == "Tomorrow":
    forecast_day_option = str(datetime.now() + timedelta(days=1))[:10]
else:
    forecast_day_option = forecast_day_option

preds = data_pred[data_pred.date == forecast_day_option]

# Iterate through list and add a marker for each volcano, color-coded by its type.
i = 0
for coordinates in geo_df_list:
    # assign a color marker for the type of volcano, Strato being the most common
    city = geo_df.city[i].lower()
    pm10 = preds[data_pred.city == city.lower()]['Mean']

    if len(pm10) == 0:
        type_color = "lightgray"
        pm10_val = "Unknown"
    else:
        pm10_val = pm10.values[0]
        
        if pm10_val < 10:
            type_color = "green"
        elif pm10_val < 20 and pm10_val >= 10:
            type_color = "blue"
        elif pm10_val < 30 and pm10_val >= 20:
            type_color = "orange"
        else:
            type_color = "purple"

    # Place the markers with the popup labels and data
    m.add_child(
        folium.Marker(
            location=coordinates,
            popup=
                "City: " + str(geo_df.city[i]) + "<br>" 
                + "Predicted PM10: " + str(pm10_val) + "<br>" 
                + "Area: " + str(geo_df['area'][i]) + "km^2<br>"
                + "Density: " + str(geo_df.density[i]) + "people/km^2<br>"
                + "Population: " + str(geo_df.no_people[i]) + "<br>",
            icon=folium.Icon(color="%s" % type_color),
        )
    )
    i = i + 1



def make_heatdata(predictions):

    heat_data = []
    for c in predictions.city:
        city_cords = geo_df[geo_df.city.apply(lambda c: c.lower()) == c].geometry.values[0]
        for _ in range(int(predictions[predictions.city == c]['Mean'].values[0])):
            heat_data.append([city_cords.xy[1][0], city_cords.xy[0][0]])
    
    # add 50 points somewhere to standardize the heatmap
    # for _ in range(50):
    #     heat_data.append([0.0, 0.0])
        
    return heat_data

heat_data = make_heatdata(preds)

plugins.HeatMap(heat_data, radius=50).add_to(m)
fig.add_child(m)

# call to render Folium map in Streamlit
with col1:

    folium_static(m)
    progress_bar.progress(80)

st.sidebar.write("Map loaded!✔️")
st.sidebar.write("-" * 36)



with col2:
    city_option = st.selectbox(
        'Select the city to view forecast plots for the whole week',
        (x for x in geo_df.city.values)
        )
    
    chart_data = data_pred[data_pred.city == city_option.lower()].drop(['city'], axis=1)
    st.line_chart(chart_data, x="date", use_container_width=True)
    
    st.markdown("<h5>💡How it's done?</h5>", unsafe_allow_html=True)
    st.write("Here, we use meta data about the cities (such as population density) together with weather forecasting data to predict the magnitude of PM10.")
    st.write("The motivation lies behind the fact, that during sudden cold weather, many people in Poland opt to use hazardous materials as fuel, such as plastics and other trash, which will cause the air quality to go down.")
    st.write("There are of course many factors contributing to that, as more developed cities will have less heating systems tied to burning fuel of an individual.")

progress_bar.progress(100)
st.button("Re-run")
