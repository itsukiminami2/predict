import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from fbprophet import Prophet
from flask import Flask, url_for, request, render_template_string

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        th, td {
            border: 1px solid black;
            padding: 8px;
        }

        table {
            border-collapse: collapse;
            font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
        }

        th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #4CAF50;
            color: white;
        }

        tr:nth-child(even){background-color: #f2f2f2;}

        tr:hover {background-color: #ddd;}
    </style>
</head>
<body>
        <h3>Predictions over the next 4 years :</h3>
            <table>
                <thead>
                    <tr>
                        <th>2018</th>
                        <th>2019</th>
                        <th>2020</th>
                        <th>2021</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{{ p[0] }}</td>
                        <td>{{ p[1] }}</td>
                        <td>{{ p[2] }}</td>
                        <td>{{ p[3] }}</td>
                    </tr>
                </tbody>
            </table>
            <br /><br />
            <h3>Nominal GDP growth : </h3>
            <img src="{{ url_for('static',filename='plot_' + country + '.png') }}"  width="800" height="400" />
</body>
</html>
'''

@app.route('/')
def predict():
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize']=(20,10)
    country = request.args['country']

    quandl.ApiConfig.api_key = "U-FFkY7wcyoehrZnXxWs"
    code = 'ODA/%s_NGDP' % country
    df = quandl.get(code)
    df = df[:-5]
    df = df.reset_index()
    df.columns = ['ds', 'y']
    
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=4, freq = 'a')
    forecast = model.predict(future)
    
    
    predictions = list(forecast[['yhat']].values.flatten())[-4:]
    return render_template_string(TEMPLATE, **{
        'country': country,
        'p': predictions
    })