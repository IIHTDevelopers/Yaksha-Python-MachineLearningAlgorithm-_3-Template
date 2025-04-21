
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Function 1: Load, clean, and prepare real loan dataset
def load_and_prepare_data(path="loan_dataset.csv"):
    """
    TODO: Implement function to load and prepare loan dataset
    - Load CSV file from the given path
    - Display loan amount statistics
    - Encode categorical columns
    - Scale features
    - Print confirmation message
    """
    # This function should fail the test without proper implementation
    # It needs to return a DataFrame with 'defaulted' column and print a specific message
    return None

# Function 2: EDA for 'loan_amount'
def explore_data(df):
    """
    TODO: Implement function to explore data
    - Analyze loan amount distribution
    - Display key statistics
    """
    # This function should fail the test if not implemented
    raise NotImplementedError("explore_data function is not implemented yet")

# Function 3: Sigmoid activation demo
def sigmoid_demo():
    """
    TODO: Implement function to demonstrate sigmoid activation
    - Calculate sigmoid of 1.5
    - Print the result
    """
    # This function should fail the test without proper implementation
    pass

# Function 4: Custom log loss cost function
def cost_function(y_true, y_pred_prob):
    """
    TODO: Implement custom log loss cost function
    - Handle edge cases with epsilon
    - Calculate binary cross-entropy
    - Return mean loss
    """
    # This function should fail the test without proper implementation
    pass

# Function 5: Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl"):
    """
    TODO: Implement function to train and evaluate model
    - Create a logistic regression model
    - Train the model with training data
    - Save the model to specified path
    - Make predictions and calculate cost
    - Print evaluation metrics and sample predictions
    """
    # This function should fail the test without proper implementation
    # It needs to create a file and print specific messages
    pass

# --------- Main Logic ---------
if __name__ == "__main__":
    df = load_and_prepare_data("loan_dataset.csv")

    explore_data(df)
    sigmoid_demo()

    X = df.drop(columns=['defaulted'])
    y = df['defaulted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)
