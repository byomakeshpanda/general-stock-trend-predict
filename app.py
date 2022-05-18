from matplotlib.pyplot import title
import streamlit as st
import datetime as date
import yfinance as yf
import pandas_datareader as data

from plotly import graph_objs as go

start = "2015-01-01"
today = date.datetime.today().strftime("%Y-%m-%d")
print(today)

st.title('Stock Prediction')
url = "https://finance.yahoo.com/"
st.write("Find the Stock picker of your stock:- [link](%s)" % url)
picker = st.text_input('Enter the stock picker from the above link by default I have taken INFY ')

data_loading = st.text("Loading Data......")

try :
    df = data.DataReader(picker , 'yahoo',start,today)
except:
    df = data.DataReader('INFY' , 'yahoo',start,today)
data_loading.text("Loading Data Completed")


st.subheader("Description about the data")
st.write("NOTE:- The data provided is in US $")
st.write(df.describe()[1:])

st.subheader('Raw data')
st.write(df.tail())

def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'] , y=df['Open'],name ='Stock_Open'))
    fig.add_trace(go.Scatter(x=df['Date'] , y=df['Close'],name ='Stock_Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()