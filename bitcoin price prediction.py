import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the bitcoin_price_data.csv dataset into a pandas DataFrame
df = pd.read_csv("bitcoin_price_data.csv")

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Split the data into training and testing sets (80/20 split)
train_size = 0.8
train_index = int(len(df) * train_size)
train_data = df[:train_index]
test_data = df[train_index:]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data[['Open']])
test_data = scaler.transform(test_data[['Open']])

# Define LSTM model architecture
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(None, 1)))
lstm_model.add(LSTM(64, activation='relu'))
lstm_model.add(Dense(1))

# Compile the model with appropriate loss function and optimizer
lstm_model.compile(loss='mean_absolute_error', optimizer='adam')

# Reshape the training data into the appropriate format for the LSTM model
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

# Train the model on the training data for a specified number of epochs
num_epochs = 50
history = lstm_model.fit(train_data, train_data, epochs=num_epochs, batch_size=16)

# Evaluate the model on the testing data using MAE as the evaluation metric
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
lstm_predicted_prices = lstm_model.predict(test_data)
lstm_predicted_prices = scaler.inverse_transform(lstm_predicted_prices)

actual_prices = scaler.inverse_transform(test_data.reshape(-1, 1))

mae = np.mean(np.abs(lstm_predicted_prices - actual_prices))
print("MAE:", mae)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(actual_prices, lstm_predicted_prices)
print("MSE:", mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Calculate Coefficient of Determination (R-squared)
r2 = r2_score(actual_prices, lstm_predicted_prices)
print("R-squared:", r2)

# Plot the actual vs. predicted Bitcoin prices for the testing data using matplotlib
plt.plot(actual_prices, label='Actual')
plt.plot(lstm_predicted_prices, label='Predicted')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_data = df[:int(len(df) * train_size)]
test_data = df[int(len(df) * train_size):]

# Create XGBoost's training and testing matrices
train_X = train_data.drop(columns=['Date', 'Close']).values
train_y = train_data['Close'].values
test_X = test_data.drop(columns=['Date', 'Close']).values
test_y = test_data['Close'].values
dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

# Define XGBoost's hyperparameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.5
}

# Train the XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the testing set
xgb_predicted_prices = xgb_model.predict(dtest)

# Evaluate the XGBoost model using MAE
mae_xgb = np.mean(np.abs(xgb_predicted_prices - test_y))
print("MAE:", mae_xgb)

# evaluate the model using mean squared error (MSE)
mse = mean_squared_error(test_y, xgb_predicted_prices)
print('MSE:', mse)

# evaluate the model using root mean squared error (RMSE)
rmse = mean_squared_error(test_y, xgb_predicted_prices, squared=False)
print('RMSE:', rmse)

# evaluate the model using coefficient of determination (R-squared)
r2 = r2_score(test_y, xgb_predicted_prices)
print('R-squared:', r2)

# Plot the actual vs. predicted Bitcoin prices for the testing data
plt.plot(test_data['Date'], test_y, label='Actual')
plt.plot(test_data['Date'], xgb_predicted_prices, label='Predicted')
plt.legend()
plt.show()

# Define the training and testing data sets
train_size = 0.8
train_index = int(len(df) * train_size)
train_data = df[:train_index]
test_data = df[train_index:]

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data[['Open']])
test_data = scaler.transform(test_data[['Open']])

# Reshape the training and testing data into the appropriate format for the models
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

# Combine the predictions into a single DataFrame
predictions_df = pd.DataFrame({
    'lstm': lstm_predicted_prices[:,0],
    'xgb': xgb_predicted_prices,
    'actual': test_data[:,0,0]
})

# Define the training and testing data for the high-level meta model
meta_train_data = predictions_df.iloc[:-1]
meta_test_data = predictions_df.iloc[-1:]

# Train the high-level meta model (Gradient Boosting Regressor)
meta_model = GradientBoostingRegressor()
meta_model.fit(meta_train_data[['lstm', 'xgb']], meta_train_data['actual'])

# Make predictions using the ensemble model
ensemble_predictions = meta_model.predict(meta_test_data[['lstm', 'xgb']])
ensemble_predictions = np.array([ensemble_predictions[0]])
ensemble_predictions = scaler.inverse_transform(ensemble_predictions.reshape(-1, 1))

# Print the ensemble prediction and the actual value
print("Ensemble prediction: ", ensemble_predictions[0])

# Calculate evaluation metrics
prediction_accuracy_percentage = meta_model.score(meta_train_data[['lstm', 'xgb']], meta_train_data['actual'])*100

# Print the evaluation metrics
print("Prediction Accuracy Percentage:", prediction_accuracy_percentage)
