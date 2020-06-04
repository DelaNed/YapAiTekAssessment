'''
Delaram Nedaei 
Description: 
Air Quality Forecasting Using LSTM-Univariate (Valnila)

References: 
[1] https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
'''


#======================== Library =============================
import numpy as np 
import tensorflow as tf 
import pandas as pd
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from Functions import split_sequence, SetTrainingSize
from sklearn.metrics import mean_squared_error

# ================== Univariate  LSTM  ========================
#------------------ Data Preparation---------------------------
# read data for training 
data = pd.read_csv('Dataset/Train/psi_df_2016_2019.csv')

# Inputs 
# this function selecting #% of the dataset
size_input = input("Set Training Size [0-70] %(Dataset Size:"+str(len(data))+"): ")
Trainingsize = SetTrainingSize(int(size_input),data)

# Select columns in data as a sequence 
national = data['national']
south = data['south']
north = data['north']
east = data['east']
central = data['central']
west = data['west']

# Create Dictionary for the columns in the dataset 
columns = {"national":national,"south":south,"north":north,"east":east,"central":central,"west":west}

# choose a number of time steps
n_steps = 24

# choose number of features
n_features = 1

# choose a number for epochs
epoch_number = 10

# LSTM Models (Output)
Models_Output = {"national":[],"south":[],"north":[],"east":[],"central":[],"west":[]}


# Main Loop for Creating LSTM Forecasting Model 
# for each column in the dataset 
for column_key in columns.keys():
    print(column_key)
    column = columns[column_key]
    # define input sequence: Train Set 
    raw_seq = column[1:Trainingsize]

    # split into samples
    X, y = split_sequence(raw_seq, n_steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # ----------------Vanilla LSTM [1]-----------------
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=epoch_number, verbose=0)
    Models_Output[column_key] = model


# =========================Prediction ===============================
print("--------------Predictions-----------------------------")
# read test data 
data_test = pd.read_csv("Dataset/Test/test.csv")

# Select columns in data as a sequence 
national_test = data_test['national']
south_test  = data_test['south']
north_test = data_test['north']
east_test = data_test['east']
central_test = data_test['central']
west_test = data_test['west']

# Create Dictionary for the columns in the dataset 
columns_test = {"national":national_test,"south":south_test,"north":north_test,"east":east_test,"central":central_test,"west":west_test}
columns_test_reshape_X = {"national":[],"south":[],"north":[],"east":[],"central":[],"west":[]}
columns_test_y =  {"national":[],"south":[],"north":[],"east":[],"central":[],"west":[]}
for column_test_key in columns_test.keys():
    print(column_test_key+"_test")
    column = columns_test[column_test_key]
    
    # define input sequence: Train Set 
    raw_seq = column

    # split into samples
    X, y = split_sequence(raw_seq, n_steps)

    columns_test_y[column_test_key] = y
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # Save X to columns_reshape_X dictionary
    columns_test_reshape_X[column_test_key] = X

# Prediction Dictionary 
Prediction_Output  = {"national":[],"south":[],"north":[],"east":[],"central":[],"west":[]}
for column_key in columns.keys():
    X = columns_test_reshape_X[column_key]
    x_input = array(X)
    yhat = model.predict(x_input, verbose=0)
    Prediction_Output[column_key] = yhat


Predictions_DataFrame = pd.DataFrame(Prediction_Output['national'])

Predictions_DataFrame.rename(columns = {0:'national'}, inplace = True) 
Predictions_DataFrame["south"] = Prediction_Output["south"]
Predictions_DataFrame["south"] = Prediction_Output["south"]
Predictions_DataFrame["north"] = Prediction_Output["north"]
Predictions_DataFrame["east"] = Prediction_Output["east"]
Predictions_DataFrame["central"] = Prediction_Output["central"]
Predictions_DataFrame["west"] = Prediction_Output["west"]

Predictions_DataFrame.to_csv("Programming/LSTM/Univariate/Predictions_Output/Predictions_output.csv")

# Output Evaluation
# MSE 
MSE = {"national":[],"south":[],"north":[],"east":[],"central":[],"west":[]}

for key in MSE.keys():
    MSE[key] = mean_squared_error(columns_test_y[key],Prediction_Output[key])
