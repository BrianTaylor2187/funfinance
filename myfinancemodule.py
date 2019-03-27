import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from pandas_datareader import data, wb
import quandl
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import column_or_1d
from bokeh.plotting import figure, output_file, show

def get_data_quandl(stock, shift=1):
    df = quandl.get(stock)
    df['returns'] = df['Value']/df['Value'].shift(shift) - 1
    df.dropna(inplace=True)
    return(df)

def get_data_yahoo(stock, start, end=dt.now(), shift=1):
    df = data.DataReader(stock, 'yahoo', start=start, end=end)
    df['Value'] = df['Close']
    df.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], axis=1, inplace=True)
    df['returns'] = df['Value']/df['Value'].shift(shift) - 1
    df.dropna(inplace=True)
    return(df)
   
def covar(df1,df2):
    df1['resid'] = df1['returns'] - df1['returns'].mean()
    df2['resid'] = df2['returns'] - df2['returns'].mean()
    covars = df1['resid']*df2['resid']
    covars.dropna()
    df1.drop('resid',axis=1)
    df2.drop('resid',axis=1)
    return(covars.sum()/covars.count())

def naive_predict_price(df):
    X = df['Value'][0:-1]
    y = df['Value'].shift(-1)[0:-1]
    X = X.values.reshape(-1,1)
    y = y.values.reshape(-1,1)
    y = column_or_1d(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    param_grid = {'C': [0.1,1,5, 10,50, 100], 'gamma': [1,0.1,0.01,0.001,0.0001]} 
    grid = GridSearchCV(SVR(),param_grid,refit=True,cv=5)
    grid.fit(X_train,y_train)
    grid_predictions = grid.predict(X_test)
    Input = X[-1].reshape(1,-1)
    Output = grid.predict(Input)
    Return = Output[0]
    df['predicted'] = grid.predict(df['Value'].values.reshape(-1,1))
    p = figure(width = 500, height = 250, x_axis_type = "datetime", sizing_mode='scale_both')
    p.line(df.index, df['predicted'], line_width = 2, color = "Orange", alpha = 0.5, legend='Predicted')
    p.line(df.index, df['Value'], line_width = 2, color = "Blue", alpha = 1, legend='Actual')
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    output_file(df.name + '.html')
    show(p)
    print('Tomorrow\'s predicted close value: ' + str(Return))
    print('Confidence: ' + str(round(100 * np.sqrt(mean_squared_error(y_test,grid_predictions))/Return, 2)) + '%')

def naive_predict_price_scaled(df):
    X = df['Value'][0:-1]
    y = df['Value'].shift(-1)[0:-1]
    X = X.values.reshape(-1,1)
    y = y.values.reshape(-1,1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    y = column_or_1d(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    param_grid = {'C': [0.1,1,5, 10,50, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 
    grid = GridSearchCV(SVR(epsilon=0.001),param_grid,refit=True,cv=5)
    grid.fit(X_train,y_train)
    grid_predictions = sc_X.inverse_transform(grid.predict(X_test))
    Input = X[-1].reshape(1,-1)
    Output = grid.predict(Input)
    Return = sc_X.inverse_transform(Output)[0]
    df['predicted'] = sc_y.inverse_transform(grid.predict(df['Value'].values.reshape(-1,1)))
    print('Tomorrow\'s predicted close value: ' + str(Return))
    print('Confidence: ' + str(np.sqrt(mean_squared_error(sc_y.inverse_transform(y_test),grid_predictions))))