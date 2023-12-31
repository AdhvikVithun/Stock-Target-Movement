

# Commented out IPython magic to ensure Python compatibility.
import pandas as PD
import numpy as np
# %matplotlib inline
import datetime as dt
import matplotlib. pyplot as plt
import matplotlib
import pandas_datareader.data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential,Model
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas_ta as ta
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

import yfinance as yf
company = 'META'
df = yf.download('META', start = '2009-01-01', end='2023-10-8')
df.to_csv('META.csv')
df

# Adding indicators
df['RSI'] = ta.rsi(df['Close'], length=15)
df['EMAF'] = ta.ema(df['Close'], length=20)
df['EMAM'] = ta.ema(df['Close'], length=100)
df['EMAS'] = ta.ema(df['Close'], length=150)

# Set up target variables and features
df['Target'] = df['Adj Close'] - df['Open']
df['Target'] = df['Target'].shift(-1)
df['TargetClass'] = [1 if val > 0 else 0 for val in df['Target']]
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

data_set = df.iloc[:, 0:11]#.values
PD.set_option('display.max_columns', None)
data_set.to_csv('All.csv')
data_set.head(20)
#print(data_set.shape)
#print(data.shape)
#print(type(data_set))

#Print the shape of Dataframe  and Check for Null Values
print("Dataframe Shape: ", df. shape)
print("Null Value Present: ", df.isnull().values.any())

#Plot the True Adj Close Value
df['Target'].plot()

#Set Target Variable
output_var = PD.DataFrame(df['Target'])
#Selecting the Features
features = ['Open', 'High', 'Low','TargetClass','Adj Close']
#'Open', 'High', 'Low'

import matplotlib.pyplot as plt

# Assuming you have 'output_var' (target variable)

# Create an index for the data points
index = range(len(output_var))

# Plot the target variable as a normal line graph in blue
plt.figure(figsize=(10, 4))  # Set figure size
plt.subplot(1, 2, 1)  # Create a subplot
plt.plot(index, output_var.values, label='Target Variable', color='blue')
plt.title("Target Variable (Line Plot)")
plt.xlabel('Data Point Index')
plt.ylabel('Scaled USD')
plt.legend()

# Plot the target variable as a dot graph in green
plt.subplot(1, 2, 2)  # Create another subplot
plt.scatter(index, output_var.values, label='Target Variable', color='green', s=10, marker='o')
plt.title("Target Variable (Dot Plot)")
plt.xlabel('Data Point Index')
plt.ylabel('Scaled USD')
plt.legend()

plt.tight_layout()  # Ensure proper layout
plt.show()

#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= PD.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()

n_splits = 10
timesplit = TimeSeriesSplit(n_splits=n_splits)

# Initialize lists to store the train and test sets
train_X_list = []
train_y_list = []
test_X_list = []
test_y_list = []

# Generate the train-test splits based on TimeSeriesSplit
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform.iloc[train_index], feature_transform.iloc[test_index]
    y_train, y_test = output_var.iloc[train_index], output_var.iloc[test_index]

    train_X_list.append(X_train)
    train_y_list.append(y_train)
    test_X_list.append(X_test)
    test_y_list.append(y_test)

train_X = PD.concat(train_X_list)
train_y = PD.concat(train_y_list)
test_X = PD.concat(test_X_list)
test_y = PD.concat(test_y_list)


# Process the data for LSTM
train_X = train_X.values.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.values.reshape(test_X.shape[0], 1, test_X.shape[1])

print(train_X.shape)
print(train_y.shape)

model_autoencoder = Sequential()
model_autoencoder.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model_autoencoder.add(LSTM(32, activation='relu', return_sequences=False))
model_autoencoder.add(RepeatVector(train_X.shape[1]))
model_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
model_autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
model_autoencoder.add(TimeDistributed(Dense(train_X.shape[2])))
model_autoencoder.compile(optimizer='adam', loss='mae')

history = model_autoencoder.fit(train_X, train_X, epochs=100, batch_size=32, validation_data=(test_X, test_X), verbose=2)

# Extract encoded representations from the middle layer of the Autoencoder
encoder_layer = Model(inputs=model_autoencoder.input, outputs=model_autoencoder.layers[2].output)
train_X1 = encoder_layer.predict(train_X)
test_X1 = encoder_layer.predict(test_X)

# Create and train an XGBoost regressor using the encoded representations
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=3,
    learning_rate=0.2,
    n_estimators=100,
    eval_metric='mae'
)



model_xgb.fit(train_X1.reshape(train_X1.shape[0], -1), train_y)

y_pred1 = model_xgb.predict(test_X1.reshape(test_X1.shape[0], -1))

# Ensure test_y is a 1D array
test_y = test_y.values.reshape(-1)

# Evaluate the XGBoost model using a regression metric (e.g., Mean Absolute Error)
mae = np.mean(np.abs(y_pred1 - test_y))
print("Mean Absolute Error (MAE):", mae)

plt.figure(figsize=(12, 6))
plt.plot( test_y, label='Actual', linewidth=2)
plt.plot( y_pred1, label='Predicted', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs. Predicted')
plt.grid(True)
plt.show()

plt.plot(test_y, label='True Values', color='blue')
plt.plot(y_pred1, label='Predicted Values', color='green')

index = range(len(test_y))

plt.scatter(index, test_y, label='True Values', color='blue', s=10, marker='s')

plt.scatter(index, y_pred1, label='Ensemble Predictions', color='green', s=10, marker='H')

n_splits = 10
timesplit = TimeSeriesSplit(n_splits=n_splits)

# Initialize lists to store the train and test sets
train_X_list = []
train_y_list = []
test_X_list = []
test_y_list = []

# Generate the train-test splits based on TimeSeriesSplit
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform.iloc[train_index], feature_transform.iloc[test_index]
    y_train, y_test = output_var.iloc[train_index], output_var.iloc[test_index]

    train_X_list.append(X_train)
    train_y_list.append(y_train)
    test_X_list.append(X_test)
    test_y_list.append(y_test)

train_X = PD.concat(train_X_list)
train_y = PD.concat(train_y_list)
test_X = PD.concat(test_X_list)
test_y = PD.concat(test_y_list)


# Process the data for LSTM
train_X = train_X.values.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.values.reshape(test_X.shape[0], 1, test_X.shape[1])

#train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
#test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2])

# Create the LSTM model
# Build the LSTM model with dropout layers
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, train_X.shape[2]), activation='tanh', return_sequences=False))
lstm.add(Dense(16))
lstm.add(Dense(8))
lstm.add(Dense(4))
lstm.add(Dense(2))
lstm.add(Dense(1))

# Compile the model
lstm.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))

# Train the model
#history = lstm.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=2)

early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss')

# Train the model with callbacks
history = lstm.fit(
    train_X, train_y,
    epochs=111,
    batch_size=22,
    verbose=2,
    shuffle=False,
    validation_data=(test_X, test_y),
    #callbacks=[early_stopping, reduce_lr]
)

test_loss = lstm.evaluate(test_X, test_y)
print(f'Test Loss: {test_loss}')

y_pred = lstm.predict(test_X)
mse = mean_squared_error(test_y, y_pred)
rmse = (mse)**0.5
print("Root Mean Squared Error:", rmse)

plt.plot(test_y, label='True Values', color='blue')
plt.plot(y_pred, label='Ensemble Predictions', color='green')

index = range(len(test_y))

plt.scatter(index, test_y, label='True Values', color='blue', s=10, marker='o')

plt.scatter(index, y_pred, label='Ensemble Predictions', color='green', s=10, marker='o')

from sklearn.svm import SVR  # Import SVR for regression

svm_regressor = SVR(kernel='rbf', C=1, gamma=0.7)
svm_regressor.fit(train_X.reshape(train_X.shape[0], -1), train_y)
svm_predictions = svm_regressor.predict(test_X.reshape(test_X.shape[0], -1))
# Plot the true 'Target' values
plt.plot(test_y, label='True Values', color='blue')
plt.plot(svm_predictions, label='SVM Predictions', color='green')

plt.title("True 'Target' vs. SVM Predictions")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

index = range(len(test_y))


plt.scatter(index, test_y, label='True Values', color='blue', s=10, marker='o')

plt.scatter(index, svm_predictions, label='Ensemble Predictions', color='green', s=10, marker='o')

plt.plot(test_y, label='True Values', color='blue')

ensemble_predictions = (y_pred1 + svm_predictions + y_pred.flatten()) / 3

ensemble_mse = mean_squared_error(test_y, ensemble_predictions)
ensemble_rmse = np.sqrt(ensemble_mse)
print("Root Mean Squared Error (Ensemble):", ensemble_rmse)

plt.plot(ensemble_predictions, label='Ensemble Predictions', color='green')

plt.title("True 'Target' vs. Ensemble Predictions")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

index = range(len(test_y))
plt.scatter(index, test_y, label='True Values', color='blue', s=10, marker='o')
plt.scatter(index, ensemble_predictions, label='Ensemble Predictions', color='green', s=10, marker='o')

index = range(len(test_y))
plt.plot(index, test_y, label='True Values', color='blue')
plt.plot(ensemble_predictions, label='Ensemble Predictions', color='green')

'''
# Plot the real data
plt.plot(test_y, label='Real Data', color='blue')

# Plot the predicted data
plt.plot(ensemble_predictions, label='Predicted Data', color='green')

plt.title("Real Data vs. Predicted Data")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()
'''

