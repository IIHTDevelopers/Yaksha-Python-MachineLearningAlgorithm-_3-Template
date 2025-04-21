import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    """
    TODO: Implement function to load and preprocess the dataset
    - Load CSV file from the given path
    - Clean column names
    - Handle missing values
    - Print confirmation message
    """
    # This function should fail the test without proper implementation
    # It needs to return a DataFrame with specific columns and print a specific message
    return None

# Function 2: Show standard deviation of price and max number of rooms
def show_key_stats(df):
    """
    TODO: Implement function to show key statistics
    - Calculate standard deviation of price
    - Find maximum number of rooms
    - Print these statistics
    """
    # This function should fail the test without proper implementation
    # It needs to print specific strings
    pass

# Function 3: Prepare data for training
def prepare_data(df, features, target):
    """
    TODO: Implement function to prepare data for training
    - Extract features and target
    - Scale features
    - Split data into training and testing sets
    - Print confirmation message
    """
    # This function should fail the test without proper implementation
    # It needs to return specific types and print a specific message
    return None, None, None, None, None

# Function 4: Train and save model
def train_and_save_model(X_train, y_train, model_path="house_price_model.pkl"):
    """
    TODO: Implement function to train and save model
    - Create a linear regression model
    - Train the model with the training data
    - Save the model to the specified path
    - Print confirmation message
    """
    # This function should fail the test without proper implementation
    # It needs to create a file and print a specific message
    return None

# Function 5: Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    TODO: Implement function to evaluate model
    - Make predictions using the model
    - Calculate mean squared error
    - Print evaluation metrics and sample predictions
    """
    # This function should fail the test without proper implementation
    # It needs to print specific strings
    pass

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['rooms', 'area', 'bathrooms', 'floors', 'age']
    target = 'price'

    df = load_and_preprocess("Housing.csv")
    show_key_stats(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, target)
    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
