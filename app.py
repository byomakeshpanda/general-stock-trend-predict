import streamlit as st
import datetime as date
import yfinance as yf
import numpy as np
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
import pandas as pd

start = "2015-01-01"
today = date.datetime.today().strftime("%Y-%m-%d")
print(today)

st.title('Stock Trend Prediction App')
url = "https://finance.yahoo.com/"
st.write("Find the Stock ticker of your stock:- [link](%s)" % url)
ticker = st.text_input('Enter the stock ticker from the above link by default I have taken INFY ')

data_loading = st.text("Loading Data......")

# try :
#     df = data.DataReader(ticker , 'yahoo',start,today)
#     df.reset_index(inplace =True)
# except:
#     df = data.DataReader('INFY' , 'yahoo',start,today)
#     df.reset_index(inplace =True)
@st.cache
def load_data(ticker):
    df = yf.download(ticker,start,today)
    df.reset_index(inplace=True)
    return df
try:
    df =load_data(ticker)
except:
    df =load_data("INFY")
data_loading.text("Loading Data Completed")


st.subheader("Description about the data")
st.write("NOTE:- The data provided is in US $ for some cases")
st.write(df.describe()[1:])

st.subheader('Latest Data')
st.write(df.tail())

def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'] , y=df['Open'],name ='Stock_Open'))
    fig.add_trace(go.Scatter(x=df['Date'] , y=df['Close'],name ='Stock_Close'))
    fig.layout.update( xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
st.subheader("Stock Price Trend of last 7 Years")
plot_raw_data()


df1 = df.reset_index()['Close']

scaler =MinMaxScaler(feature_range=(0,1))
df2 = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df2)*0.70) #70% of the data is used for training
test_size = len(df2) - training_size
train_data,test_data = df2[0:training_size,:],df2[training_size:len(df2),:1]

def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step=150
X_train,y_train =create_dataset(train_data,time_step)
X_test,y_test= create_dataset(test_data,time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model =Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(150,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64,verbose=1)

train_predict=model.predict(X_train)
test_predict= model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

look_back=150
trainPredictPlot = np.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict

train_plot=pd.DataFrame(trainPredictPlot)
train_plot.rename(columns={0:'Values'},inplace=True)

testPredictPlot = np.empty_like(df2)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict

test_plot=pd.DataFrame(testPredictPlot)
test_plot.rename(columns={0:'Values'},inplace=True)

# fig = plt.figure(figsize=(10,5))
# plt.plot(scaler.inverse_transform(df2))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# st.pyplot(fig)

st.subheader("Actual Data Trained Data and Predicted Data")
fig =go.Figure()
fig.add_trace(go.Scatter(x=df['Date'] , y=df['Close'],name ='Actual Data'))
fig.add_trace(go.Scatter(x=df['Date'] , y=train_plot['Values'],name ='Trained Data'))
fig.add_trace(go.Scatter(x=df['Date'] , y=test_plot['Values'],name ='Tested Data'))
fig.layout.update(xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

x_input = test_data[len(test_data)-150:].reshape(1,-1)
temp_input = list(x_input)
temp_input=temp_input[0].tolist()

# lst_output=[]
# n_steps = 150
# i=0
# while(i<30):
#     if (len(temp_input)>150):
#         x_input = np.array(temp_input[1:])
#         x_input=x_input.reshape(1,-1)
#         x_input=x_input.reshape((1,n_steps,1))
#         yhat = model.predict(x_input, verbose=0)
#         lst_output.extend(yhat.tolist())
#         i=i+1
#     else:
#         x_input = x_input.reshape((1,n_steps,1))
#         yhat = model.predict(x_input,verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         lst_output.extend(yhat.tolist())
#         i=i+1

from numpy import array

lst_output=[]
n_steps=150
i=0
while(i<30):
    
    if(len(temp_input)>150):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

# print(lst_output)


day_new=np.arange(1,151)
day_pred =np.arange(151,181)

st.subheader("Next 30 days predicted trend")
fig = plt.figure(figsize=(10,5))
plt.plot(day_new,scaler.inverse_transform(df2[len(df2)-150:]),label="Current Trend",color="blue")
plt.plot(day_pred,scaler.inverse_transform(lst_output),label="Predicted Trend",color="green")
st.pyplot(fig)
