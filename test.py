import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./WindPower/Turbine_Data.csv')

df_selected = df[['Unnamed: 0', 'ActivePower', 'WindSpeed']]
#start of dataset is 2017-12-31 00:00:00+00:00
print(df_selected.head())
#end of dataset is 2020-03-30 23:50:00+00:00 
print(df_selected.tail())

#we select the input features
input_df = df_selected[['ActivePower', 'WindSpeed']]

#generate timestamp for the observations mimicking the "unnamed:0" feature
rng = pd.date_range('2017-12-31', periods=118224, freq='10T')
time_df = pd.DataFrame(rng)
#fill in missing values with zero using the fillforward function
input_df = input_df.fillna(0).astype(float)
#concatenate both the timestamp range and filled in features
input_df = pd.concat((time_df, input_df), axis=1)
#set up index
input_df = input_df.set_index(0)
#select a subset of data period from second half of original dataset, to ensure we have better quality signal (see completness comment of msno matrix plot above in the code)
input_df = input_df.loc['2019-12-17':]
input_df.head()

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an LSTM input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

# Number of lags (steps back in 10min intervals) to use for models
lag = 360
# Steps in future to forecast (steps in 10min intervals)
n_ahead = 144
# ratio of observations for training from total series
train_share = 0.8
# training epochs
epochs = 20
# Batch size , which is the number of samples of lags
batch_size = 512
# Learning rate
lr = 0.001
# The features for the modeling 
features_final = ['ActivePower','WindSpeed']

# Subseting only the needed columns
ts = input_df[features_final]

#Scaling data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(ts)
ts_scaled = scaler.transform(ts)

# Creating the X and Y for training, the formula is set up to assume the target Y is the left most column = target_index=0
X, Y = create_X_Y(ts_scaled, lag=lag, n_ahead=n_ahead)

# Spliting into train and test sets
Xtrain, Ytrain = X[0:int(X.shape[0] * train_share)
                   ], Y[0:int(X.shape[0] * train_share)]
Xtest, Ytest = X[int(X.shape[0] * train_share)
                     :], Y[int(X.shape[0] * train_share):]
