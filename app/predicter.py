import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import EarlyStopping
import os
import preprocessing  # Ensure this module is correctly imported

class StockPredictor:
    def __init__(self, data_path: str, step_size: int = 1):
        self.data_path = data_path
        self.step_size = step_size
        self.model = None
        self.scaler = None
        self.dataset = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None

    def load_data(self):
        """Load and reverse the dataset."""
        self.dataset = pd.read_csv(self.data_path, usecols=[1, 2, 3, 4])  # Select OHLC data
        self.dataset = self.dataset.reindex(index=self.dataset.index[::-1])  # Reverse the dataset

    def preprocess_data(self):
        """Preprocess dataset and split it into training and test sets."""
        # Calculate the average of OHLC data (open, high, low, close)
        OHLC_avg = self.dataset.mean(axis=1)

        # Reshape and normalize the data
        OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg), 1))
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        OHLC_avg = self.scaler.fit_transform(OHLC_avg)

        # Split data into training and testing sets
        train_size = int(len(OHLC_avg) * 0.75)  # 75% for training, 25% for testing
        self.train_data, self.test_data = OHLC_avg[:train_size], OHLC_avg[train_size:]

        # Prepare training and testing datasets using the preprocessing module
        self.trainX, self.trainY = preprocessing.new_dataset(self.train_data, self.step_size)
        self.testX, self.testY = preprocessing.new_dataset(self.test_data, self.step_size)

        # Reshape the data to be suitable for the LSTM model
        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

    def build_model(self):
        """Build and compile the LSTM model."""
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(1, self.step_size), return_sequences=True))
        self.model.add(Dropout(0.2))  # Dropout to prevent overfitting
        self.model.add(LSTM(32))
        self.model.add(Dropout(0.2))  # Dropout to prevent overfitting
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))

        # Compile the model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self, epochs: int = 5, batch_size: int = 1):
        """Train the LSTM model with early stopping and validation split."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            self.trainX, self.trainY, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,  # Use 20% of the data for validation
            verbose=2,
            callbacks=[early_stopping]
        )
        
        self.plot_loss_history(history)

    def plot_loss_history(self, history):
        """Plot the training and validation loss over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('static/training_loss_plot.png')
        plt.close()

    def save_model(self, model_filename: str = 'stock_price_model.pkl', scaler_filename: str = 'scaler.pkl'):
        """Save the trained model and scaler."""
        # Save the model using Pickle
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        print(f"Model saved to {model_filename}")

        # Save the scaler using Pickle
        with open(scaler_filename, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        print(f"Scaler saved to {scaler_filename}")

    def load_model(self, model_filename: str = 'stock_price_model.pkl', scaler_filename: str = 'scaler.pkl'):
        """Load the model and scaler from files."""
        # Load the model using Pickle
        with open(model_filename, 'rb') as model_file:
            self.model = pickle.load(model_file)
        print(f"Model loaded from {model_filename}")

        # Load the scaler using Pickle
        with open(scaler_filename, 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
        print(f"Scaler loaded from {scaler_filename}")

    def evaluate_model(self):
        """Evaluate the model's performance and generate predictions."""
        # Make predictions for training and test sets
        trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)

        # De-normalize predictions
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([self.trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([self.testY])

        # Calculate RMSE (Root Mean Square Error)
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print(f'Train RMSE: {trainScore:.2f}')

        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print(f'Test RMSE: {testScore:.2f}')

        # Plot the results
        self.plot_results(trainPredict, testPredict)

        return trainPredict, testPredict, trainY, testY

    def plot_results(self, trainPredict, testPredict):
        """Plot the original dataset, training predictions, and test predictions."""
        OHLC_avg = self.dataset.mean(axis=1).values
        trainPredictPlot = np.empty_like(OHLC_avg)
        testPredictPlot = np.empty_like(OHLC_avg)

        trainPredictPlot[:] = np.nan
        testPredictPlot[:] = np.nan

        # Plot training predictions
        trainPredictPlot[:len(trainPredict)] = trainPredict[:, 0]

        # Plot test predictions
        start_idx = len(trainPredict)
        end_idx = start_idx + len(testPredict)
        testPredictPlot[start_idx:end_idx] = testPredict[:, 0]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(OHLC_avg, 'g', label='Original Dataset')
        plt.plot(trainPredictPlot, 'r', label='Training Set')
        plt.plot(testPredictPlot, 'b', label='Predicted Stock Price / Test Set')
        plt.legend(loc='upper right')
        plt.xlabel('Time in Days')
        plt.ylabel('OHLC Value of Apple Stocks')

        # Save the plot to a file in the static directory
        plot_filename = 'static/prediction_plot.png'
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory

        # Return the plot filename for FastAPI to serve
        return plot_filename

    def predict_future(self):
        """Predict the future stock price based on the last available value."""
        # Predict the last day's stock price
        last_val = self.model.predict(self.testX[-1].reshape(1, 1, self.testX.shape[2]))
        last_val_scaled = last_val / last_val  # Normalize the prediction

        # Predict the next day's stock price
        next_val = self.model.predict(np.reshape(last_val_scaled, (1, 1, 1)))

        print(f"Last Day Value: {last_val[0]}")
        print(f"Next Day Value: {last_val * next_val}")
        return last_val, next_val
