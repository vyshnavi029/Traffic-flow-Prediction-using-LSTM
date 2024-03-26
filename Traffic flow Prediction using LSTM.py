#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
#Loading the required datset
data = pd.read_csv('D:\Machine Learning\ML_ASSIGNMENT1_121AD0029\Dataset.csv')


# In[14]:


# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

#Preprocessing the given dataset
# Converting the 'Hour' column to a datetime format
data['Hour'] = pd.to_datetime(data['Hour'], format='%m-%d-%Y %H:%M')

# Specifying the features whichever wanted to use for prediction
features = ['Lane 1 Flow (Veh/Hour)', 'Lane 1 Speed (mph)', 'Lane 1 Occ (%)']

# Normalizing the selected features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Defining the sequence length (lookback window) and number of features
sequence_length = 30
num_features = len(features)

# Creating sequences for input data and their corresponding targets
sequences = []
targets = []

for i in range(len(data) - sequence_length):
    sequences.append(data[features].values[i:i + sequence_length])
    targets.append(data[features].values[i + sequence_length])

sequences = np.array(sequences)
targets = np.array(targets)

# Spliting the data into training and testing sets
split_ratio = 0.8
split_index = int(len(sequences) * split_ratio)

X_train, X_test = sequences[:split_index], sequences[split_index:]
y_train, y_test = targets[:split_index], targets[split_index:]

# Here we are defining a function to build and compile the LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, num_features), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build, train, and predict for short-term traffic prediction
model_short_term = build_lstm_model()
model_short_term.fit(X_train, y_train, epochs=60, batch_size=32, verbose=2)

train_predictions_short_term = model_short_term.predict(X_train)
test_predictions_short_term = model_short_term.predict(X_test)

# Inverse Transformations for Short-Term Predictions
train_predictions_short_term = scaler.inverse_transform(train_predictions_short_term)
test_predictions_short_term = scaler.inverse_transform(test_predictions_short_term)
y_train_short_term = scaler.inverse_transform(y_train)
y_test_short_term = scaler.inverse_transform(y_test)



# Visualizing the results for Short-Term
# Short-Term Train Data
plt.figure(figsize=(10, 8))
plt.title('Short-Term Train Data vs. Predictions')
plt.xlabel('Time')
plt.ylabel('Traffic Data')
plt.plot(y_train_short_term[:, 0], label='Actual Flow', color ='blue')
plt.plot(train_predictions_short_term[:, 0], label='Predicted Flow', color = 'green')
plt.legend()
plt.show()

# Short-Term Test Data
plt.figure(figsize=(10, 8))
plt.title('Short-Term Test Data vs. Predictions')
plt.xlabel('Time')
plt.ylabel('Traffic Data')
plt.plot(y_test_short_term[:, 0], label='Actual Flow', color ='blue')
plt.plot(test_predictions_short_term[:, 0], label='Predicted Flow', color = 'green')
plt.legend()
plt.show()



# Evaluate the Model for Short-Term and Long-Term Predictions
# Calculating Mean Squared Error and Root Mean Squared Error for short-term and long-term predictions
mse_train_short_term = mean_squared_error(y_train_short_term, train_predictions_short_term)
rmse_train_short_term = np.sqrt(mse_train_short_term)
mse_test_short_term = mean_squared_error(y_test_short_term, test_predictions_short_term)
rmse_test_short_term = np.sqrt(mse_test_short_term)
r2_test_short_term = r2_score(y_test_short_term, test_predictions_short_term)



print("Results for the Short-Term")
print(f"Train MSE: {mse_train_short_term:.4f}, \nRMSE: {rmse_train_short_term:.4f}")
print(f"Test MSE: {mse_test_short_term:.4f}, \nRMSE: {rmse_test_short_term:.4f},  R^2: {r2_test_short_term:.4f}")


# In[21]:


data1 = pd.read_csv('D:\Machine Learning\ML_ASSIGNMENT1_121AD0029\longterm_dataset.csv')

data1['Day'] = pd.to_datetime(data1['Day'], format='%m/%d/%Y %H:%M')

# Defining the date range for long-term prediction (Taken this dates based on the top 5 head & last 5 tail values)
start_date_long_term = pd.Timestamp('2019-10-23')
end_date_long_term = pd.Timestamp('2023-10-26')

# Filter data for the desired date range for long-term prediction
data1 = data1[(data1['Day'] >= start_date_long_term) & (data1['Day'] <= end_date_long_term)]
features = ['Lane 1 Flow (Veh/Day)', 'Lane 1 Speed (mph)', 'Lane 1 Occ (%)']
# Normalizing the selected features for long-term prediction
scaler_long_term = MinMaxScaler()
data1[features] = scaler_long_term.fit_transform(data1[features])

# Creating sequences for long-term input data and their corresponding targets
sequences_long_term = []
targets_long_term = []

for i in range(len(data1) - sequence_length):
    sequences_long_term.append(data1[features].values[i:i + sequence_length])
    targets_long_term.append(data1[features].values[i + sequence_length])

sequences_long_term = np.array(sequences_long_term)
targets_long_term = np.array(targets_long_term)

# Spliting the data into training and testing sets for long-term prediction
split_index_long_term = int(len(sequences_long_term) * split_ratio)

X_train_long_term, X_test_long_term = sequences_long_term[:split_index_long_term], sequences_long_term[split_index_long_term:]
y_train_long_term, y_test_long_term = targets_long_term[:split_index_long_term], targets_long_term[split_index_long_term:]

# Build, train, and predict for long-term traffic prediction
model_long_term = build_lstm_model()
model_long_term.fit(X_train_long_term, y_train_long_term, epochs=80, batch_size=32, verbose=2)

train_predictions_long_term = model_long_term.predict(X_train_long_term)
test_predictions_long_term = model_long_term.predict(X_test_long_term)

# Inverse transformations for long-term predictions
train_predictions_long_term = scaler_long_term.inverse_transform(train_predictions_long_term)
test_predictions_long_term = scaler_long_term.inverse_transform(test_predictions_long_term)
y_train_long_term = scaler_long_term.inverse_transform(y_train_long_term)
y_test_long_term = scaler_long_term.inverse_transform(y_test_long_term)


# Long-Term Train Data
plt.figure(figsize=(10, 8))
plt.title('Long-Term Train Data vs. Predictions')
plt.xlabel('Time')
plt.ylabel('Traffic Data')
plt.plot(y_train_long_term[:, 0], label='Actual Flow', color ='blue')
plt.plot(train_predictions_long_term[:, 0], label='Predicted Flow', color = 'green')
plt.legend()
plt.show()

# Long-Term Test Data
plt.figure(figsize=(10, 8))
plt.title('Long-Term Test Data vs. Predictions')
plt.xlabel('Time')
plt.ylabel('Traffic Data')
plt.plot(y_test_long_term[:, 0], label='Actual Flow', color ='blue')
plt.plot(test_predictions_long_term[:, 0], label='Predicted Flow', color = 'green')
plt.legend()
plt.show()


mse_train_long_term = mean_squared_error(y_train_long_term, train_predictions_long_term)
rmse_train_long_term = np.sqrt(mse_train_long_term)
mse_test_long_term = mean_squared_error(y_test_long_term, test_predictions_long_term)
rmse_test_long_term = np.sqrt(mse_test_long_term)
r2_test_long_term = r2_score(y_test_long_term, test_predictions_long_term)


print("Results for the Long-Term")
print(f"Train MSE: {mse_train_long_term:.4f},\n RMSE: {rmse_train_long_term:.4f}")
print(f"Test MSE: {mse_test_long_term:.4f},\n RMSE: {rmse_test_long_term:.4f},\n R^2: {r2_test_long_term:.4f}")


# In[ ]:




