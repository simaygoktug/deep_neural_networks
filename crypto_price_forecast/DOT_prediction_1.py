import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the dataset
df = pd.read_csv('./DOT/DOT_All_Time.csv')
df = df.set_index("Date")

# Select the 'Close' column as the target variable
data = df.filter(['Close']).values

# Split the dataset into training and testing data
train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
test_data = data[train_size:, :]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Create the training data and labels
x_train = []
y_train = []

for i in range(60, len(train_data_scaled)):
    x_train.append(train_data_scaled[i-60:i, 0])
    y_train.append(train_data_scaled[i, 0])

# Convert the training data and labels into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the training data to be 3-dimensional
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Create the testing data and labels
x_test = []
y_test = []

for i in range(60, len(test_data_scaled)):
    x_test.append(test_data_scaled[i-60:i, 0])
    y_test.append(test_data_scaled[i, 0])

# Convert the testing data and labels into numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the testing data to be 3-dimensional
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions on the testing data
predictions = model.predict(x_test)

# Unscale the data
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(y_test, color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.legend()
plt.show()

# Predict the price after 30 days
last_60_days = data[-60:]
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

print("The predicted price for DOT after 30 days is: ", predicted_price)

###########################################################################################

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate the model
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Unscale the data
train_predict = scaler.inverse_transform(train_predict)
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train_unscaled, train_predict)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_unscaled, train_predict)
train_r2 = r2_score(y_train_unscaled, train_predict)

test_mse = mean_squared_error(y_test_unscaled, test_predict)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_unscaled, test_predict)
test_r2 = r2_score(y_test_unscaled, test_predict)

print("Training set evaluation metrics:")
print("MSE: ", train_mse)
print("RMSE: ", train_rmse)
print("MAE: ", train_mae)
print("R-squared score: ", train_r2)
print("\nTesting set evaluation metrics:")
print("MSE: ", test_mse)
print("RMSE: ", test_rmse)
print("MAE: ", test_mae)
print("R-squared score: ", test_r2)
