import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Load synthetic fish disease dataset
def load_fish_disease_data():
    """
    Load the fish disease dataset from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing fish disease data with 1000 records
    """
    # TODO: Load the fish disease dataset from 'fish_disease_data.csv'
    # TODO: Take only the first 1000 records using .head(1000)
    # TODO: Print loading information messages
    # TODO: Return the DataFrame
    pass
    return pd.DataFrame()  # Return empty DataFrame as placeholder

# 2. EDA Function to count fish with age > 1 year
def perform_eda_on_age(df):
    """
    Perform exploratory data analysis on the Age column.
    
    Args:
        df (pd.DataFrame): DataFrame containing fish disease data
    """
    # TODO: Check if 'Age' column exists in the DataFrame
    # TODO: Count fish with age > 1 year
    # return count 
    return count
    pass 

# 3. Preprocess data
def preprocess_fish_data(df):
    """
    Preprocess the fish disease data.
    
    Args:
        df (pd.DataFrame): DataFrame containing fish disease data
        
    Returns:
        tuple: (X, y, df_encoded)
            - X (pd.DataFrame): Features DataFrame
            - y (pd.Series): Target Series
            - df_encoded (pd.DataFrame): Encoded DataFrame
    """
    # TODO: Use pd.get_dummies() to encode categorical variables
    # TODO: Check if 'Disease_Status_Healthy' column exists after encoding
    # TODO: Separate features (X) and target (y)
    # TODO: Return X, y, and the encoded DataFrame
    pass
    return pd.DataFrame(), pd.Series(), pd.DataFrame()  # Return empty placeholders

# 4. Split the data
def split_fish_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features DataFrame
        y (pd.Series): Target Series
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Use train_test_split to split the data with random_state=42
    # TODO: Return X_train, X_test, y_train, y_test
    pass
    # Return None values to ensure test fails until properly implemented
    return None, None, None, None

# 5. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    """
    Create and train a Decision Tree model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        DecisionTreeClassifier: Trained model
    """
    # TODO: Create a DecisionTreeClassifier with random_state=42
    # TODO: Train the model on X_train and y_train
    # TODO: Print model creation and training information messages
    # TODO: Return the trained model
    pass
    return DecisionTreeClassifier()  # Return untrained model as placeholder

# 6. Calculate entropy
def calculate_entropy(y):
    """
    Calculate the entropy of the target variable.
    
    Args:
        y (pd.Series): Target Series
        
    Returns:
        float: Entropy value
    """
    # TODO: Calculate value counts with normalize=True
    # TODO: Calculate entropy using the formula: -sum(p * log2(p))
    # TODO: Return the entropy value
    pass
    return 0.0  # Return placeholder value

# 7. Predict new fish data from JSON
def check_new_data_from_json(model, df_encoded, json_file="fish_data.json"):
    """
    Predict disease status for new fish data from a JSON file.
    
    Args:
        model (DecisionTreeClassifier): Trained model
        df_encoded (pd.DataFrame): Encoded DataFrame
        json_file (str): Path to JSON file with new fish data
        
    Returns:
        int: Prediction (1 for Healthy, 0 for Diseased)
    """
    # TODO: Import json module
    # TODO: Load fish data from the JSON file
    # TODO: Load original dataset
    # TODO: Create a temporary DataFrame with the new fish data
    # TODO: Combine with original data and encode
    # TODO: Extract features for the new fish
    # TODO: Make prediction using the model
    # TODO: Return the prediction (1 for Healthy, 0 for Diseased)
    pass
    # Return None to ensure test fails until properly implemented
    return int()
