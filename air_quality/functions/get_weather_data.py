import requests
import pandas as pd
from pandas import json_normalize
from dotenv import load_dotenv
import os
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_weather_data(save_to_csv=True):
    df = pd.DataFrame()

    df_meta = pd.read_excel('data_poland/meta.xlsx')

    cities = df_meta['city']

    load_dotenv()
    secret = os.getenv('WEATHER_API_KEY')

    api = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + \
        str(cities[0]) + \
        "?unitGroup=metric&key=" + str(secret) + "&contentType=json"
    # print(api)
    response = requests.get(f"{api}")

    if response.status_code == 200:
        #print("successful fetch")
        r = response.json()
        df = json_normalize(r)

        for i in range(1, len(cities)):
            api = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + \
                str(cities[i]) + \
                "?unitGroup=metric&key=" + str(secret) + "&contentType=json"
            response = requests.get(f"{api}")

            if response.status_code == 200:
                #print("successful fetch")
                r = response.json()
                df = pd.concat([df, json_normalize(r)], ignore_index=True)
    else:
        # Return error message
        return 1

    df_data = pd.DataFrame([[df['days'][0][0]['datetime'], df['address'][0], df['days'][0][0]['tempmax'], df['days'][0][0]['tempmin'],
                             df['days'][0][0]['temp'], df['days'][0][0]['feelslikemax'], df[
                                 'days'][0][0]['feelslikemax'], df['days'][0][0]['feelslike'],
                             df['days'][0][0]['dew'], df['days'][0][0]['humidity'], df['days'][0][0]['precip'], df['days'][0][0]['precipprob'],
                             df['days'][0][0]['precipcover'], df['days'][0][0]['snow'], df['days'][0][0]['snowdepth'], df['days'][0][0]['windgust'],
                             df['days'][0][0]['windspeed'], df['days'][0][0]['winddir'], df[
                                 'days'][0][0]['cloudcover'], df['days'][0][0]['visibility'],
                             df['days'][0][0]['solarradiation'], df['days'][0][0]['solarenergy'], df['days'][0][0]['uvindex'], df['days'][0][0]['conditions']]],
                           columns=['date', 'city', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
                                    'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
                                    'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
                                    'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
                                    'conditions'])

    df_temp = pd.DataFrame()

    for i in range(0, len(df)):
        for k in range(0, 7):
            if not (i == 0 and k == 0):
                df_temp = pd.DataFrame([[df['days'][i][k]['datetime'], df['address'][i], df['days'][i][k]['tempmax'], df['days'][i][k]['tempmin'],
                                         df['days'][i][k]['temp'], df['days'][i][k]['feelslikemax'], df[
                                             'days'][i][k]['feelslikemax'], df['days'][i][k]['feelslike'],
                                         df['days'][i][k]['dew'], df['days'][i][k]['humidity'], df[
                                             'days'][i][k]['precip'], df['days'][i][k]['precipprob'],
                                         df['days'][i][k]['precipcover'], df['days'][i][k]['snow'], df[
                                             'days'][i][k]['snowdepth'], df['days'][i][k]['windgust'],
                                         df['days'][i][k]['windspeed'], df['days'][i][k]['winddir'], df[
                                             'days'][i][k]['cloudcover'], df['days'][i][k]['visibility'],
                                         df['days'][i][k]['solarradiation'], df['days'][i][k]['solarenergy'], df['days'][i][k]['uvindex'], df['days'][i][k]['conditions']]],
                                       #df['days'][0][0][''], df['days'][0][0][''], df['days'][0][0][''], df['days'][0][0][''],
                                       columns=['date', 'city', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
                                                'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
                                                'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
                                                'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
                                                'conditions'])  # , 'area', 'density', 'no_people'])

            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    df_data['city'] = df_data['city'].str.lower()
    df_data['conditions'] = df_data['conditions'].str.lower()

    df_meta['city'] = df_meta['city'].str.lower()
    df_meta_temp = df_meta
    df_meta_temp.drop(columns='city_id')

    df_data = df_data.join(df_meta_temp.set_index('city'), on='city')


    scaler = StandardScaler()

    cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax',
            'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
            'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
            'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex']

    le = LabelEncoder()
    df_data['conditions'] = le.fit_transform(df_data['conditions'])

    df_data[cols] = scaler.fit_transform(df_data[cols])


    if save_to_csv:
        df_data.to_csv(f'weather_files/{str(datetime.datetime.now())[:10]}.csv', index=False)

    return df_data

