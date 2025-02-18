from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Base model for all the models
Base = declarative_base()

class StockPrediction(Base):
    __tablename__ = "stock_predictions"  # The name of the table in the database

    # Columns
    id = Column(Integer, primary_key=True, index=True)  # Primary key and index for faster lookups
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)  # Indexed for better search performance
    predicted_stock_price = Column(Float, nullable=False)  # Make sure this is required (non-nullable)
    actual_stock_price = Column(Float, nullable=True)  # Optional field for actual stock price
    model_version = Column(String, default="v1.0")  # Store model version or training version

    # New fields
    user_last_values = Column(String, nullable=True)  # Allow NULL for user input (to be optional)
    train_rmse = Column(Float, nullable=True)  # Allow NULL for training RMSE (to be optional)
    test_rmse = Column(Float, nullable=True)  # Allow NULL for testing RMSE (to be optional)
