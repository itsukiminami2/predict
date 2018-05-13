import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from fbprophet import Prophet
from flask import Flask, url_for

app = Flask(__name__)

@app.route('/<country>')
def predict_gdp(country, self):
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize']=(20,10)

    quandl.ApiConfig.api_key = "U-FFkY7wcyoehrZnXxWs"
    code = 'ODA/%s_NGDP' % country
    df = quandl.get(code)
    df = df.reset_index()
    df.columns = ['ds', 'y']
    
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=2, freq = 'a')
    forecast = model.predict(future)

    df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)

    df_plot = df.join(forecast[['yhat']], how = 'outer')
    df_plot = df_plot.reset_index()
    df_plot.columns = ['year', 'current gdp', 'predicted gdp']
    df_plot.set_index('year', inplace=True)

    df_plot.plot()
    fig = plt.gcf()
    plt.title('Nominal GDP Growth')
    plt.ylabel('GDP (billions of US$)')
    fig.savefig('static\plot_%s.png' % country, dpi=100)

    return '<img src=' + url_for('static',filename='plot_%s.png' % country) + ' width=800 height=400 />'

if __name__ == '__main__':
    app.run()