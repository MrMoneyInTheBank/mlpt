import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Prepare the DataFrame
sp500_data = yf.download('^GSPC', start='2010-01-01', end='2020-12-31')
russell_data = yf.download('^RUT', start='2010-01-01', end='2020-12-31')

df = pd.DataFrame({
    'SP500_Close': sp500_data['Close'],
    'Russell_Close': russell_data['Close']
})
df['Price_Ratio'] = df['SP500_Close'] / df['Russell_Close']

# Add moving averages as additional features
df['MA50'] = df['Price_Ratio'].rolling(window=50).mean()
df['MA200'] = df['Price_Ratio'].rolling(window=200).mean()
df.dropna(inplace=True)  # Drop rows with NaN values resulting from moving averages

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Price_Ratio', 'MA50', 'MA200']])

# Function to create sequences with multiple features
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(sequence_length, len(data)):
        x = data[i-sequence_length:i]
        y = data[i][0]  # Assuming the target is the first feature (Price_Ratio)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dropout(0.2))
model.add(Dense(units=1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(np.hstack((y_pred, np.zeros(y_pred.shape), np.zeros(y_pred.shape))))
y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros(y_test.reshape(-1, 1).shape), np.zeros(y_test.reshape(-1, 1).shape))))

# Plotting the results
plt.figure(figsize=(14,5))
plt.plot(y_test_rescaled[:, 0], color='red', label='Real Price Ratio')
plt.plot(y_pred_rescaled[:, 0], color='blue', label='Predicted Price Ratio')
plt.title('Price Ratio Prediction')
plt.xlabel('Time')
plt.ylabel('Price Ratio')
plt.legend()
plt.show()

# Generating signals
signals_df = pd.DataFrame(index=df.index[-len(y_test):])
signals_df['Real'] = y_test_rescaled[:, 0]
signals_df['Predicted'] = y_pred_rescaled[:, 0]
signals_df['Signal'] = 0
signals_df.loc[signals_df['Predicted'] > signals_df['Real'], 'Signal'] = 1
signals_df.loc[signals_df['Predicted'] < signals_df['Real'], 'Signal'] = -1

# Backtesting
initial_capital = 100000
positions = initial_capital * signals_df['Signal']
portfolio = positions.diff()
portfolio[0] = positions[0]
cumulative_returns = (portfolio / initial_capital).cumsum() + 1
final_return = cumulative_returns[-1] - 1

plt.figure(figsize=(14,5))
plt.plot(cumulative_returns, color='green', label='Cumulative Returns')
plt.title('Backtest Results')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

print(f"Final Return: {final_return * 100:.2f}%")
