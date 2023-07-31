# Import required libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import SimpleRNN

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

# Define the LSTM model architecture
lstm_model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(units=128, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

# Build an RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=256, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
rnn_model.add(Dropout(0.3))
rnn_model.add(SimpleRNN(units=128, return_sequences = False))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(units=1))

# Compile the model
rnn_model.compile(optimizer='adam', loss='mae')
rnn_model.summary()

# Define callbacks
early_stopping = EarlyStopping(patience=10, verbose=1)

# For LSTM model
model_checkpoint_lstm = ModelCheckpoint('best_lstm_model.h5', save_best_only=True, verbose=1)

# For SimpleRNN model
model_checkpoint_rnn = ModelCheckpoint('best_rnn_model.h5', save_best_only=True, verbose=1)

# Train LSTM model
#history_lstm = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint_lstm], verbose=1)

# Train SimpleRNN model
history_rnn = rnn_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint_rnn], verbose=1)

# Load best saved LSTM model
lstm_model.load_weights('best_lstm_model.h5')

# Evaluate models on testing set
y_test_pred_lstm = lstm_model.predict(X_test).flatten()
test_loss_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
r2_lstm = r2_score(y_test, y_test_pred_lstm)
print("MAE on the testing set: {:.2f}".format(test_loss_lstm))
print("R^2 score on the testing set: {:.2f}".format(r2_lstm))

rnn_model.load_weights('best_rnn_model.h5')

y_test_pred_rnn = rnn_model.predict(X_test).flatten()
test_loss_rnn = mean_absolute_error(y_test, y_test_pred_rnn)
r2_rnn = r2_score(y_test, y_test_pred_rnn)
print("MAE on the testing set: {:.2f}".format(test_loss_rnn))
print("R^2 score on the testing set: {:.2f}".format(r2_rnn))

fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_lstm, color='red')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('LSTM Model: Predicted vs. Actual TEC (Testing Set)')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_rnn, color='red')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('LSTM Model: Predicted vs. Actual TEC (Testing Set)')
plt.show()

# Plot training and validation losses for LSTM and SimpleRNN
# plt.figure(figsize=(10, 5))
# plt.plot(history_lstm.history['loss'],color='Red', label='LSTM Training Loss')
# plt.plot(history_lstm.history['val_loss'],color='Blue', linestyle='--', label='LSTM Validation Loss')
# plt.plot(history_rnn.history['loss'],color='Brown', label='RNN Training Loss')
# plt.plot(history_rnn.history['val_loss'],color='Black', linestyle='--', label='RNN Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.savefig('fig6.png')
