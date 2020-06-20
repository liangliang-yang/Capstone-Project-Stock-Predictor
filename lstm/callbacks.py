import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import plotly.graph_objs as go

import flask
import pandas as pd
import numpy as np
import colorlover as cl
import yfinance as yf
import re

from app import app
from utils import bbands

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

import warnings
warnings.filterwarnings('ignore')


colorscale = cl.scales['9']['qual']['Paired']

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# define business day
B_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def download_stock_to_df(symbol, start, end):
    """
    Get current stocks data from yahoo fiance and save to dataframe

    Params:
        symbol: stock to pull data
        start: start date of pulled data
        end: end date of pulled data

    Return:
        dataframe of stock within specified date range
    """
    df_stock=yf.download(symbol,start,end,progress=False)
    df_stock.reset_index(level=0, inplace=True)
    return df_stock

def generate_train_sequence(train_data, n_steps):
    """
    Generate sequence array for train data,
    inclduing both X and y
    """
    X = []
    y = []
    N = len(train_data)
    for i in range(N):
        end_index = i + n_steps # find the end of this sequence
        # check if we are out of index
        if end_index > N-1:
            break
        seq_x = train_data[i:end_index]
        seq_y = train_data[end_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def generate_test_sequence(test_data, n_steps):
    """
    Generate sequence array for test data,
    inclduing only X
    """
    X = []
    N = len(test_data)
    for i in range(N):
        end_index = i + n_steps # find the end of this sequence
        # check if we are out of index
        if end_index > N-1:
            break
        seq_x = test_data[i:end_index]
        X.append(seq_x)
    return np.array(X)

def predict_future_price(df, lookback_days, forcast_days, model, sc):
    """
    Prediction of future stock price
    """

    prediction_index_list = []
    prediction_date_list = []
    prediction_price_list = []
    prediction_price_scaled_list = []

    train_data = df.copy(deep=True) # init train data
    prediction_start_index = train_data.index[-1]+1
    prediction_start_date = train_data['Date'].iloc[-1]+1*B_DAY # need to use bussiness day

    for i in range(forcast_days):

        X_scaled = train_data[['Scaled_Close']][-lookback_days:].values

        # using MinMaxScaler to normalize the price
        # X_scaled = sc.transform(X)
        X_scaled = X_scaled.reshape((1, lookback_days, 1))

        # prediction the scaled price with model
        prediction_price_scaled = model.predict(X_scaled)

        # transform back to the normal price
        prediction_price = sc.inverse_transform(prediction_price_scaled)[0][0]

        # append prediction date, index and price
        prediction_index = prediction_start_index + i
        prediction_date = prediction_start_date + i*B_DAY

        prediction_index_list.append(prediction_index)
        prediction_date_list.append(prediction_date)
        prediction_price_list.append(prediction_price)
        prediction_price_scaled_list.append(prediction_price_scaled[0][0])

        # update the train_data
        train_data.loc[prediction_index, 'Date'] = prediction_date # update train_data
        train_data.loc[prediction_index, 'Adj Close'] = prediction_price # update train_data
        train_data.loc[prediction_index, 'Scaled_Close'] = prediction_price_scaled[0][0] # update train_data


    # create the forcast_dataframe
    forcast_dataframe = pd.DataFrame({
        'index' : prediction_index_list,
        'Date' : prediction_date_list,
        'Adj Close' : prediction_price_list,
        'Scaled_Close' : prediction_price_scaled_list
    })

    forcast_dataframe = forcast_dataframe.set_index('index', drop=True)
    return forcast_dataframe


def init_callbacks(symbol):
    """
    Function to init all callbacks in the dash app

    Parameters
    ----------
    df: DataFrame

    Returns:
    ---------
    None
    """

    # @app.callback(
    #     dash.dependencies.Output('start-date-output-container', 'children'),
    #     [dash.dependencies.Input('start-date-picker-range', 'date')])
    # def update_start_date(date):
    #     if date is not None:
    #         date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    #         date_string = date.strftime('%B %d, %Y')
    #         return 'You have selected {} for stock start date'.format(date_string)
    #
    # @app.callback(
    #     dash.dependencies.Output('end-date-output-container', 'children'),
    #     [dash.dependencies.Input('end-date-picker-range', 'date')])
    # def update_end_date(date):
    #     if date is not None:
    #         date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    #         date_string = date.strftime('%B %d, %Y')
    #         return 'You have selected {} for stock end date'.format(date_string)

    # @app.callback(
    #     dash.dependencies.Output(component_id='graph', component_property='children'),
    #     [
    #         dash.dependencies.Input(component_id='stock-symbol-input', component_property='value')
    #     ]
    # )

    @app.callback(
        dash.dependencies.Output('output','children'),
        [
            dash.dependencies.Input('reset_button','n_clicks')
        ]
    )
    def update(reset):
        if reset > 0:
            reset = 0
            return 'all clear'



    @app.callback(
        dash.dependencies.Output(component_id='graph', component_property='children'),
        [
            dash.dependencies.Input(component_id='stock-symbol-input', component_property='value'),
            dash.dependencies.Input('start-date-picker', 'date'),
            dash.dependencies.Input('end-date-picker', 'date'),
            dash.dependencies.Input('input_button', 'n_clicks')
        ]
    )

    def update_graph(symbol, start_date, end_date, n_clicks):

        # df = download_stock_to_df(symbol, '2015-01-01', '2019-01-01')

        if n_clicks < 1:
            return "Click Run Model after date selections to begin."

        start_date = str(start_date)
        end_date = str(end_date)
        print(symbol, start_date, end_date)

        df = download_stock_to_df(symbol, start_date, end_date)
        df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # scale all data
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0.01,0.99))
        df['Scaled_Close'] = sc.fit_transform(df[['Adj Close']])

        # split into train and test/validation
        from sklearn.model_selection import train_test_split
        dataset_train , dataset_test = train_test_split(df, train_size=0.9, test_size=0.1, shuffle=False)

        train_set = dataset_train[['Scaled_Close']].values
        test_set = dataset_test[['Scaled_Close']].values

        lookback_days = 15

        # train
        X_train, y_train = generate_train_sequence(train_set, lookback_days)
        # X_train, y_train = generate_train_sequence(train_set_scaled, lookback_days)

        # init model
        model = Sequential()
        model.add(LSTM(units=lookback_days, activation='relu', input_shape=(lookback_days,1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        num_epochs = 100
        model.fit(X_train,y_train,epochs=num_epochs,batch_size=32)

        # test
        test_set_extend = dataset_train[['Scaled_Close']][-lookback_days:].append(dataset_test[['Scaled_Close']]).values
        # test_set_extend = dataset_train[['Adj Close']][-lookback_days:].append(dataset_test[['Adj Close']]).values
        # test_set_extend_scaled = sc.transform(test_set_extend)

        X_test = generate_test_sequence(test_set_extend, lookback_days)

        # prediction = model.predict(X_test)
        prediction_scaled = model.predict(X_test)
        prediction = sc.inverse_transform(prediction_scaled)

        train_set = train_set.reshape((-1))
        test_set = test_set.reshape((-1))
        prediction = prediction.reshape((-1))

        dataset_test['Prediction'] = prediction

        # forcast
        # data_price = df[['Adj Close']].values
        # data_index = df[['Adj Close']].index

        forcast_days = 5
        # re-train the model with full data set_index, removed as for now
        # full_set = df[['Scaled_Close']].values
        # X_train_full, y_train_full = generate_train_sequence(full_set, lookback_days)
        # model.fit(X_train_full,y_train_full,epochs=num_epochs,batch_size=32)

        dataset_forcast = predict_future_price(df, lookback_days, forcast_days, model, sc)
        # forcast_set = dataset_forcast['Adj Close'].values
        # forcast_index = dataset_forcast.index


        trace1 = go.Scatter(
            x = dataset_train['Date'],
            y = dataset_train['Adj Close'],
            mode = 'lines',
            name = 'Train-set'
        )
        trace2 = go.Scatter(
            x = dataset_test['Date'],
            y = dataset_test['Prediction'],
            mode = 'lines',
            name = 'Test-set-Prediction'
        )
        trace3 = go.Scatter(
            x = dataset_test['Date'],
            y = dataset_test['Adj Close'],
            mode='lines',
            name = 'Test-set-Ground_Truth'
        )
        trace4 = go.Scatter(
        x = dataset_forcast['Date'],
        y = dataset_forcast['Adj Close'],
        # mode='lines+markers',
        line = dict(color='red', width=2, dash='dot'),
        name = 'Forcast'
    )
        layout = go.Layout(
            title = "AAPL Stock",
            xaxis = {'title' : "Date"},
            yaxis = {'title' : "Adj Close ($)"}
        )
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        fig.update_layout(xaxis_range=[dataset_train['Date'].iloc[0], dataset_forcast['Date'].iloc[-1]+100*B_DAY])

        graph = dcc.Graph(
            id='stock_prediction_graph',
            figure=fig
        )



        # stock_data = {
        #     'x': df['Date'],
        #     'y': df['Adj Close'],
        #     'line': {'width': 2, 'color': 'red'},
        #     'name': ticker + ' Adj Close Price',
        #     'legendgroup': ticker,
        # }
        #
        # bb_bands = bbands(df['Adj Close'])
        # bollinger_traces = [{
        #     'x': df['Date'], 'y': y,
        #     'type': 'scatter',
        #     'mode': 'lines',
        #     'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
        #     # 'line': {'width': 1, 'color': 'blue'},
        #     'hoverinfo': 'none',
        #     # 'legendgroup': ticker,
        #     # 'showlegend': True if i == 0 else False,
        #     'showlegend': True,
        #     'name': 'upper Bollinger Bands' if i == 0 else 'lower Bollinger Bands',
        # } for i, y in enumerate(bb_bands)]
        #
        # graph = dcc.Graph(
        #     id=ticker,
        #     figure={
        #         'data': [stock_data] + bollinger_traces,
        #         'layout': {
        #             'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
        #             'legend': {'x': 0}
        #         }
        #     }
        # )

        return graph


    # def update_graph(tickers):
    #     graphs = []
    #
    #     if not tickers:
    #         graphs.append(html.H3(
    #             "Select a stock ticker.",
    #             style={'marginTop': 20, 'marginBottom': 20}
    #         ))
    #     else:
    #         for i, ticker in enumerate(tickers):
    #
    #             # dff = df[df['Stock'] == ticker]
    #             df = download_stock_to_df(ticker, '2015-01-01', '2019-01-01')
    #
    #             candlestick = {
    #                 'x': df['Date'],
    #                 'open': df['Open'],
    #                 'high': df['High'],
    #                 'low': df['Low'],
    #                 'close': df['Close'],
    #                 'type': 'candlestick',
    #                 'name': ticker,
    #                 'legendgroup': ticker,
    #                 'increasing': {'line': {'color': colorscale[0]}},
    #                 'decreasing': {'line': {'color': colorscale[1]}}
    #             }
    #             bb_bands = bbands(df.Close)
    #             bollinger_traces = [{
    #                 'x': df['Date'], 'y': y,
    #                 'type': 'scatter', 'mode': 'lines',
    #                 'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
    #                 'hoverinfo': 'none',
    #                 'legendgroup': ticker,
    #                 'showlegend': True if i == 0 else False,
    #                 'name': '{} - bollinger bands'.format(ticker)
    #             } for i, y in enumerate(bb_bands)]
    #             graphs.append(dcc.Graph(
    #                 id=ticker,
    #                 figure={
    #                     'data': [candlestick] + bollinger_traces,
    #                     'layout': {
    #                         'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
    #                         'legend': {'x': 0}
    #                     }
    #                 }
    #             ))
    #
    #     return graphs
