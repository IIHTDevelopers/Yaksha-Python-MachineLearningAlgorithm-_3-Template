import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Function 1: Load and inspect data
def load_data(path="loan_dataset.csv"):
    """
    Load loan dataset from CSV file and print basic statistics.
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing loan data
    """
    # TODO: Load the CSV file using pd.read_csv()
    # TODO: Print loan amount statistics (mean and max) with proper formatting
    # TODO: Return the DataFrame
    pass
    return pd.DataFrame()  # Return empty DataFrame as placeholder

# Function 2: Basic EDA — count people living in rented homes
def explore_home_ownership(df):
    """
    Count the number of people living in rented homes.
    
    Args:
        df (pd.DataFrame): DataFrame containing loan data
        
    Returns:
        int: Count of people living in rented homes
    """
    # TODO: Check if 'home_ownership' column exists in the DataFrame
    # TODO: Count records where home_ownership is 'RENT'
    # TODO: Print the count with appropriate message
    # TODO: Return the count
    pass
    return None  # Return None to ensure test fails until properly implemented

# Function 3: Encode and scale features
def prepare_data(df):
    """
    Encode categorical variables and scale numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing loan data
        
    Returns:
        pd.DataFrame: Processed DataFrame with encoded and scaled features
    """
    # TODO: Use LabelEncoder to encode 'term' and 'home_ownership' columns if they are object type
    # TODO: Create a StandardScaler
    # TODO: Identify features to scale (all columns except 'defaulted')
    # TODO: Scale the features
    # TODO: Print confirmation message
    # TODO: Return the processed DataFrame
    pass
    return pd.DataFrame()  # Return empty DataFrame as placeholder

# Function 4: Sigmoid function demo
def sigmoid_demo():
    """
    Demonstrate the sigmoid function with a specific input value.
    
    Returns:
        float: Sigmoid value for z=1.5
    """
    # TODO: Set z to 1.5
    # TODO: Calculate sigmoid using the formula: 1 / (1 + np.exp(-z))
    # TODO: Print the result with appropriate message
    # TODO: Return the sigmoid value
    pass
    return None  # Return None to ensure test fails until properly implemented

# Function 5: Train and evaluate logistic regression model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl"):
    """
    Train a logistic regression model and evaluate it.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        path (str): Path to save the model
        
    Returns:
        dict: Dictionary containing model, predictions, and probabilities
    """
    # TODO: Create a LogisticRegression model with max_iter=1000
    # TODO: Train the model on X_train and y_train
    # TODO: Save the model using joblib.dump()
    # TODO: Print confirmation message
    # TODO: Make predictions on X_test
    # TODO: Calculate prediction probabilities
    # TODO: Print sample predictions
    # TODO: Return dictionary with model, predictions, and probabilities
    pass
    # Return None to ensure test fails until properly implemented
    return None
