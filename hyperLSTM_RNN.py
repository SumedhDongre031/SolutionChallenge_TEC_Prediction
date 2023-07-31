import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

# Load the dataset
df = pd.read_excel('TEST.xlsx')

# Split the data into training, testing, and validation sets 60|20|20
X_train, X_test, y_train, y_test = train_test_split(df.drop('TEC', axis=1), df['TEC'], test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Define LSTM model architecture
def create_lstm_model(units=64, dropout=0.2, recurrent_dropout=0.2):
    model = keras.Sequential([
        layers.LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(10, 1)),
        layers.Dense(units=1)
    ])
    model.compile(loss='mae', optimizer='adam')
    return model

# Define SimpleRNN model architecture
def create_rnn_model(units=64, dropout=0.2, recurrent_dropout=0.2):
    model = keras.Sequential([
        layers.SimpleRNN(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(10, 1)),
        layers.Dense(units=1)
    ])
    model.compile(loss='mae', optimizer='adam')
    return model

# Define hyperparameters to search over for each model
lstm_param_grid = {
    'units': [64, 128],
    'dropout': [0.2, 0.3],
    'recurrent_dropout': [0.2, 0.3]
}

rnn_param_grid = {
    'units': [64, 128],
    'dropout': [0.2, 0.3],
    'recurrent_dropout': [0.2, 0.3]
}

# Create KerasRegressor objects for each model
#lstm_model = KerasRegressor(build_fn=create_lstm_model, epochs=100, batch_size=64, verbose=0)
rnn_model = KerasRegressor(build_fn=create_rnn_model, epochs=100, batch_size=64, verbose=0)

# Define RandomizedSearchCV objects for each model
#lstm_random = RandomizedSearchCV(estimator=lstm_model, param_distributions=lstm_param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)
rnn_random = RandomizedSearchCV(estimator=rnn_model, param_distributions=rnn_param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)

# Fit the models with early stopping callback
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

#lstm_random.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es])
rnn_random.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es])

# Evaluate models on testing set
#y_test_pred_lstm = lstm_random.predict(X_test).flatten()
y_test_pred_rnn = rnn_random.predict(X_test).flatten()

# Compute MAE and R2 scores
#test_loss_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
test_loss_rnn = mean_absolute_error(y_test, y_test_pred_rnn)
#r2_lstm = r2_score(y_test, y_test_pred_lstm)
r2_rnn = r2_score(y_test, y_test_pred_rnn)

# Print the MAE and R2 scores
#print("Test loss for LSTM model:", test_loss_lstm)
#print("R2 score for LSTM model:", r2_lstm)
print("Test loss for SimpleRNN model:", test_loss_rnn)
print("R2 score for SimpleRNN model:", r2_rnn)