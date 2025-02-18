Stock Price Prediction API

Project Overview:

Stock Price Prediction API is a FastAPI application that uses an LSTM (Long Short-Term Memory) model to predict the stock price of Apple Inc. based on historical data. This API allows you to train a model on historical stock data, make predictions, visualize results, and evaluate the model's performance. It also integrates with an SQLite database to store and manage predictions.

Features:

Model Training: Train the LSTM model on historical stock data.
Stock Price Prediction: Predict future stock prices based on recent values.
Model Evaluation: Evaluate the model’s performance with RMSE (Root Mean Square Error).
Visualizations: Generate and view plots of predicted stock prices.
Database Integration: Store predictions in a SQLite database.
Technologies Used
FastAPI: Web framework to create the API.
TensorFlow / Keras: For building and training the LSTM model.
Scikit-learn: For data preprocessing (scaling, splitting data).
SQLite: Lightweight database to store predictions.
Matplotlib: For generating visualizations of the predictions.

Installation
Prerequisites
Python 3.7 or higher

Step-by-Step Installation
1)Clone the repository
2)Create a virtual environment:
python -m venv venv
3)Activate the virtual environment:
venv\Scripts\activate
4)Install dependencies:
pip install -r requirements.txt

Running the API
To run the FastAPI application:
uvicorn main:app --reload

The app will be accessible at http://127.0.0.1:8000.

Database Integration:
Predictions are stored in an SQLite database, which can be queried or modified as needed.

To see the predictions in the database, use:

SELECT * FROM stock_predictions;

Model Training
Model Details: The model uses an LSTM architecture to predict the stock price. The training data consists of OHLC (Open, High, Low, Close) values.
Training Parameters: You can specify the number of epochs and batch size during training.

This project was developed as a practice project at SYSTEMS LIMITED to explore stock price prediction using machine learning and web development.
