# Stock Predictor with LSTM and Dash

This project goal is to demonstrate how to use LSTM (Long Short Term Memory) networks to predict the stock price. In addition, a simple web application was built for the stock predictor.

## Notebook and Report

In the repository, you can find two notebooks, which demonstrate how to predict stock price step by step with LSTM network. Apartment from the LSTM model, a Simple Moving Average model was provided to compare the performance.

Besides the two notebooks, you can find the detailed report for the work.

## Dash Web Application

In addition to the LSTM stock prediction model building and analysis, an interactive web application was also created. In order to run the web application, please follow these steps:
* Git clone the repository
  + `git clone https://github.com/liangliang-yang/Capstone-Project-Stock-Predictor.git`

* Create the conda env with the `requirement.yml`
  + `cd Capstone-Project-Stock-Predictor`
  + `conda env create -f environment.yml`
  + `conda activate ML`

* Run the web application by:
  + `cd lstm`
  + `python index.py`
  + When the web server is on, go to http://127.0.0.1:8080/
  + Choose the stock symbol from the dropdown lists (some stock examples are provided.) Choose the start_date and end_date, and the click `Run Model` button. The whole model running process will take about 10 minutes, you can see the progress in the terminal. After that, you can see the interactive plot for stock price (including train test and forcast).
  + ![Screenshot of the web application](/figures/LSTM.PNG)


## Prerequisites
The code was written in Python, and uses:

* Pandas and Numpy: for data manipulation
* Matplotlib: for data visualisation
* Keras: for building and training the model
* sklearn: for data scaling
* Dash Plotly: for interactive web application creation
