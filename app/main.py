from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from predicter import StockPredictor  # Ensure this is correctly importing your StockPredictor class

# Import async database-related modules
from database import get_db, init_db, shutdown_db  # Database connection and init functions
from models import StockPrediction
from sqlalchemy.ext.asyncio import AsyncSession  # Ensure AsyncSession is imported for async DB operations

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html as the homepage
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "../templates", "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())

# Initialize the Stock Predictor with the file path for the stock data
predictor = StockPredictor(data_path="../data/apple_share_price.csv")

# Pydantic model for the prediction request
class PredictionRequest(BaseModel):
    last_values: list

# Train the model
@app.post("/train_model")
async def train_model(request: dict):
    epochs = int(request.get('epochs', 5))  # Default to 5 if not provided
    batch_size = int(request.get('batch_size', 32))  # Default to 32 if not provided

    # Load, preprocess, build, and train the model
    predictor.load_data()  # Load the stock data
    predictor.preprocess_data()  # Preprocess the data
    predictor.build_model()  # Build the model
    predictor.train_model(epochs=epochs, batch_size=batch_size)  # Train the model

    # Save model and scaler after training
    predictor.save_model()  # This saves both the model and the scaler

    return JSONResponse(content={"message": "Model training completed successfully!"})

# Load model and scaler before predicting stock price
@app.on_event("startup")
async def load_model_and_scaler():
    model_path = "stock_price_model.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        predictor.load_model(model_path, scaler_path)
        print("Model and scaler loaded successfully.")
    else:
        print("Model or scaler not found. Please train the model first.")
    
    try:
        # Initialize the database (create tables if not already present)
        await init_db()
        print("Database initialized!")

        # Get the absolute path of the database
        db_path = os.path.abspath("./stock_predicter.db")  # Replace with your actual database filename
        print(f"SQLite database path: {db_path}")  # This will print the path to the console

    except Exception as e:
        print(f"Error initializing database: {e}")

# Predict stock price for a given set of last values and store in the database
@app.post("/predict_stock_price")
async def predict_stock_price(request: PredictionRequest, db: AsyncSession = Depends(get_db)):
    last_values = request.last_values  # Get the last known stock values from the user
    
    # Ensure the values are numeric before transforming
    try:
        last_values = np.array(last_values, dtype=float)  # Convert to float array
    except ValueError:
        return JSONResponse(content={"error": "All values in 'last_values' must be numeric."}, status_code=400)

    # Check if the scaler is loaded before using it
    if predictor.scaler is None:
        return JSONResponse(content={"error": "Scaler is not loaded. Please train the model first."}, status_code=500)

    # Reshape and scale the data for the model
    last_values = last_values.reshape(-1, 1)  # Reshape to 2D for scaling
    last_values_scaled = predictor.scaler.transform(last_values)  # Transform using the scaler

    # Reshape for LSTM input
    last_values_scaled = np.reshape(last_values_scaled, (1, 1, len(last_values_scaled)))  # Reshape for LSTM input

    try:
        # Make the prediction
        predicted_stock_price = predictor.model.predict(last_values_scaled)
        
        # Denormalize the predicted value
        predicted_stock_price = predictor.scaler.inverse_transform(predicted_stock_price)
        
        # Convert to native Python float before returning
        predicted_stock_price = float(predicted_stock_price[0][0])

        # Calculate RMSE for both training and test sets
        trainPredict, testPredict, trainY, testY = predictor.evaluate_model()

        trainRMSE = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        testRMSE = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        # Convert last values to string to store them in the database
        user_last_values_str = ",".join(map(str, last_values.flatten()))

        # Store the prediction in the database with RMSE values
        new_prediction = StockPrediction(
            predicted_stock_price=predicted_stock_price,
            user_last_values=user_last_values_str,  # Save the user's last input values as a string
            train_rmse=trainRMSE,                   # Store the train RMSE
            test_rmse=testRMSE                     # Store the test RMSE
        )

        db.add(new_prediction)
        await db.commit()  # Ensure async commit
        await db.refresh(new_prediction)  # Refresh after commit to get the latest data

        return JSONResponse(content={
            "predicted_stock_price": predicted_stock_price,
            "prediction_id": new_prediction.id,
            "train_rmse": trainRMSE,
            "test_rmse": testRMSE
        })
    
    except Exception as e:
        return JSONResponse(content={"error": f"Error during prediction: {str(e)}"}, status_code=500)

# Generate and return the plot URL
@app.get("/generate_plot")
async def generate_plot():
    try:
        plot_file_path = "static/stock_price_plot.png"
        
        if predictor.dataset is None:
            return JSONResponse(content={"error": "Model is not evaluated yet. Please train the model first."}, status_code=500)

        trainPredict, testPredict, trainY, testY = predictor.evaluate_model()

        if len(trainPredict) == 0 or len(testPredict) == 0:
            return JSONResponse(content={"error": "Prediction data is empty. Ensure the model is properly trained."}, status_code=500)

        OHLC_avg = predictor.dataset.mean(axis=1).values
        trainPredictPlot = np.empty_like(OHLC_avg)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[1:len(trainPredict) + 1] = trainPredict[:, 0]

        testPredictPlot = np.empty_like(OHLC_avg)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict):len(trainPredict) + len(testPredict)] = testPredict[:, 0]

        plt.figure(figsize=(10,6))
        plt.plot(OHLC_avg, 'g', label='Original Dataset')
        plt.plot(trainPredictPlot, 'r', label='Training Set')
        plt.plot(testPredictPlot, 'b', label='Predicted Stock Price / Test Set')
        plt.legend(loc='upper right')
        plt.xlabel('Time in Days')
        plt.ylabel('OHLC Value of Apple Stocks')

        plt.savefig(plot_file_path)
        plt.close()

        return JSONResponse(content={"plot_url": f"/static/stock_price_plot.png"})

    except Exception as e:
        print(f"Error generating plot: {e}")
        return JSONResponse(content={"error": f"Error generating plot: {str(e)}"}, status_code=500)

# Endpoint to get model metrics (RMSE)
@app.get("/model_metrics")
async def get_model_metrics():
    trainPredict, testPredict, trainY, testY = predictor.evaluate_model()

    trainRMSE = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testRMSE = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return JSONResponse(content={"trainRMSE": trainRMSE, "testRMSE": testRMSE})

# Database connection on startup and shutdown
@app.on_event("shutdown")
async def shutdown():
    # Shutdown and disconnect the database
    await shutdown_db()
    print("Database disconnected!")
