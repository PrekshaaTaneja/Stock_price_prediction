import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader( 'This app is created to forecast the stock market price of the selected company.')
# add an image from online resource
st. image( "https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg" )


# take input from the user of app about the start and end date
#sidebar
st.sidebar. header( 'Select the parameters from below')

start_date = st.sidebar.date_input('Start date',date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))

ticker_list =["AAPL", "MSFT", "GOOG", "GOOGL", "FBI", "TSLA", "NVDA", "ADBE","PYPL","INTC", "CMCSA", "NFLX","PEP"]
ticker=st.sidebar. selectbox('Select the company', ticker_list)

data = yf.download(ticker, start=start_date,end=end_date)

data.insert(0,"Date", data. index, True)
data.reset_index(drop=True, inplace=True)
st.write( 'Data from',start_date,'to',end_date)
st.write(data)

#plot the data
st.header( 'Dat Visualization')
st.subheader( 'Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig=px.line(data,x='Date',y=data.columns,title='Closing price of the stock')
st.plotly_chart(fig)

column= st.selectbox('Select the column to be used for forecasting',data.columns[1:])

data = data[['Date',column]]
st.write( "Selected data")
st.write(data)

st.header('Is data stationary?')
st.write(adfuller(data[column])[1]<0.05)

st.header( 'Decomposition of the data')
decomposition= seasonal_decompose(data [column] ,model=' additive' ,period=12)
st. write( decomposition.plot())

st.write("## Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend, title='Trend',width=1000, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal, title='Seasonality',width=1000, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid, title='Residuals',width=1000, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='red',line_dash='dot'))

p = st.slider( 'Select the value of p',0,5,2)
d = st.slider( 'Select the value of d',0,5,1)
q=st.slider( 'Select the value of q',0,5,2)
seasonal_order = st.number_input( 'Select the value of seasonal p', 0, 24, 12)

model = sm.tsa.statespace.SARIMAX( data [column] ,order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model =model.fit(disp=1)

# st.header( 'Model Summary' )
# st.write(model.summary())
# st.write("--")
st.write("## Forecasting the data")

forecast_period = st.number_input( 'Select the number of days to forecast',1, 365, 10)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean
# st.write(predictions)

predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0,"Date", predictions.index,True)
predictions.reset_index(drop=True, inplace=True)
st.write("Predictions",predictions)
st.write("Actual Data",data)
st.write("--")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data [column], mode='lines', name='Actual', line=dict (color=' blue' )))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions ["predicted_mean"], mode='lines', name='Predicted', line=dict (color='red' )))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date' ,yaxis_title=' Price ', width=1200, height=400)
st.plotly_chart(fig)

show_plots = False
if st.button( 'Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data [column] ,title='Actual', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"] ,title='Actual', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        show_plots=True
    else:
        show_plots= False

hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plot = False

st.write("--")