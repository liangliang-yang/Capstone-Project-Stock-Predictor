import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime
from datetime import datetime as dt

import pandas as pd
import numpy as np

from app import app
from callbacks import init_callbacks

today = datetime.date.today()

def run_server():
    """
    Main function to run the server
    """

    symbol_options = [
        {'label': 'AAPL', 'value': 'AAPL'},
        {'label': 'MSFT', 'value': 'MSFT'},
        {'label': 'AMZN', 'value': 'AMZN'}
    ]

    app.layout = html.Div([
        html.Div([
            html.H2('LSTM Stock Predictor',
                    style={'display': 'inline',
                           'float': 'left',
                           'font-size': '2.65em',
                           'margin-left': '7px',
                           'font-weight': 'bolder',
                           'font-family': 'Product Sans',
                           'color': "rgba(117, 117, 117, 0.95)",
                           'margin-top': '20px',
                           'margin-bottom': '0'
                           }),
        ]),
        dcc.Dropdown(
            id='stock-symbol-input',
            # options=[{'label': s, 'value': s}
            #          for s in symbols],
            options = symbol_options,
            value='AAPL'
        ),

        # html.Div([
        #     dcc.DatePickerRange(
        #         id='stock-date-picker-range',
        #         min_date_allowed=dt(2000, 1, 1),
        #         max_date_allowed=today,
        #         initial_visible_month=dt(2015, 1, 1),
        #         end_date=today
        #     ),
        #     # html.Div(id='output-container-date-picker-range')
        # ]),

        html.Div([
            dcc.DatePickerSingle(
                id='start-date-picker',
                min_date_allowed=dt(1990, 1, 1),
                max_date_allowed=today,
                initial_visible_month=dt(2000, 1, 1),
                # date=dt(2015, 1, 1)
            ),
            # html.Div(id='start-date-output-container')
        ]),


        html.Div([
            dcc.DatePickerSingle(
                id='end-date-picker',
                min_date_allowed=dt(1990, 1, 1),
                max_date_allowed=today,
                initial_visible_month=dt(2020, 6, 1),
                # date=today
            ),
        ]),

        html.Button('Run Model', id='input_button', n_clicks=0),
        # html.Button('Reset',id='reset_button', n_clicks=0),

        html.Div(id='graph')
    ], className="container")


    # initialize callbacks
    init_callbacks('AAPL')

    # app.run_server(debug=True, port=8080, threaded=False)
    app.run_server(debug=True, port=8080)

if __name__ == '__main__':
    run_server()
