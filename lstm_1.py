'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

df = pd.read_excel('TEST.xlsx')
print(df.columns)

#Split the data into training, testing, and validation sets 60|20|20
X_train, X_test, y_train, y_test = train_test_split(df.drop('TEC', axis=1), df['TEC'], test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Build an LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(units=128))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

keras.utils.plot_model(lstm_model, to_file='plot_lstm.png', show_layer_names=True)
print("PLOT HAS BEEN GENERATED, PLEASE CHECK YOUR DIRECTORY")

# Fit the model to the training data
history_lstm = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), verbose=1)
#Print the MSE and R^2 score for the LSTM model
print("======================================")
print("LSTM Model:")
print("Training set:")


train_loss_lstm = mean_absolute_error(y_train, y_train_pred_lstm)
print("MAE: ", train_loss_lstm)
print("R^2 score: ", r2_score(y_train, y_train_pred_lstm))
print("\nTesting set:")

y_train_pred_lstm_filename = "y_train_pred_lstm.txt"
create_file(y_train_pred_lstm_filename, y_train_pred_lstm)
#print("LSTM Test Predictions: ", y_test_pred_lstm)

test_loss_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
print("MAE: ", test_loss_lstm)
print("R^2 score: ", r2_score(y_test, y_test_pred_lstm))

y_test_pred_lstm_filename = "y_test_pred_lstm.txt"
create_file(y_test_pred_lstm_filename, y_test_pred_lstm)
#print("LSTM Test Predictions: ", y_test_pred_lstm)

print("\nValidation set:")
val_loss_lstm = mean_absolute_error(y_val, y_val_pred_lstm)
print("MAE: ", val_loss_lstm)
print("R^2 score: ", r2_score(y_val, y_val_pred_lstm))

# Plotting the results for LSTM
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred_lstm, color='red')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('LSTM Model: Predicted vs. Actual TEC (Testing Set)')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
plt.savefig('fig4.png')
'''
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model_checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
#history = lstm_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint], verbose=1)

# Load the best saved model
lstm_model.load_weights('best_lstm_model.h5')

# Evaluate the model on the testing set
y_test_pred = lstm_model.predict(X_test).flatten()
test_loss = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("MAE on the testing set: {:.2f}".format(test_loss))
print("R^2 score on the testing set: {:.2f}".format(r2))

# Plot the predicted vs. actual TEC for the testing set
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred, color='red')
ax.plot(np.arange(0, 55, 1), np.arange(0, 55, 1), color='black', linestyle='-')
ax.set_xlabel('Actual TEC')
ax.set_ylabel('Predicted TEC')
ax.set_title('LSTM Model: Predicted vs. Actual TEC (Testing Set)')
plt.show()
