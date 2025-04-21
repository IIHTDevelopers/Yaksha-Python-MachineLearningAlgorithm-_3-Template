# House Price and Loan Default Prediction Implementation Guide

This document provides detailed instructions for implementing the required functions in `House.py` and `loan.py` files to pass the functional tests in `test_functional.py`. The implementation should follow the skeleton code structure with proper functionality.

## Dataset Information

### Housing Dataset (`Housing.csv`)
The housing dataset contains information about house properties with the following features:
- `rooms`: Number of rooms in the house
- `area`: Area of the house in square feet
- `bathrooms`: Number of bathrooms
- `floors`: Number of floors
- `age`: Age of the house in years
- `price`: Price of the house (target variable)

### Loan Dataset (`loan_dataset.csv`)
The loan dataset contains information about loans with the following features:
- `loan_amount`: Amount of the loan
- `term`: Loan term (36 months or 60 months)
- `credit_score`: Credit score of the borrower
- `employment_length`: Employment length in years
- `home_ownership`: Type of home ownership (OWN, MORTGAGE, RENT)
- `annual_income`: Annual income of the borrower
- `defaulted`: Whether the loan defaulted (0 = no default, 1 = default) (target variable)

## House.py Implementation Requirements

### 1. `load_and_preprocess(path)`
**Purpose**: Load and preprocess the housing dataset.
**Requirements**:
- Load the CSV file from the given path
- Clean column names (convert to lowercase and strip whitespace)
- Handle missing values (drop rows with missing values)
- Print a confirmation message: "✅ Data loaded and cleaned."
- Return the preprocessed DataFrame

### 2. `show_key_stats(df)`
**Purpose**: Display key statistics about the housing data.
**Requirements**:
- Calculate the standard deviation of the price column
- Find the maximum number of rooms
- Print these statistics in a formatted way:
  - "📊 Standard Deviation of Price: $[value]"
  - "🛏️ Maximum Number of Rooms: [value]"

### 3. `prepare_data(df, features, target)`
**Purpose**: Prepare the data for model training.
**Requirements**:
- Extract features (X) and target (y) from the DataFrame
- Scale the features using StandardScaler
- Split the data into training and testing sets (80% train, 20% test) with random_state=42
- Print a confirmation message: "📊 Data prepared and split."
- Return X_train, X_test, y_train, y_test, and the scaler object

### 4. `train_and_save_model(X_train, y_train, model_path="house_price_model.pkl")`
**Purpose**: Train a linear regression model and save it.
**Requirements**:
- Create a LinearRegression model
- Train the model with the training data
- Save the model to the specified path using joblib.dump()
- Print a confirmation message: "✅ Model trained and saved to '[model_path]'"
- Return the trained model

### 5. `evaluate_model(model, X_test, y_test)`
**Purpose**: Evaluate the model's performance.
**Requirements**:
- Make predictions using the model on the test data
- Calculate the mean squared error
- Print the evaluation metrics:
  - "📉 Mean Squared Error: [value]"
  - "🔍 Sample Predictions: [first 10 predictions]"

## loan.py Implementation Requirements

### 1. `load_and_prepare_data(path="loan_dataset.csv")`
**Purpose**: Load and preprocess the loan dataset.
**Requirements**:
- Load the CSV file from the given path
- Display loan amount statistics:
  - "📊 Loan Amount - Mean: [value], Max: [value]"
- Encode categorical columns ('term', 'home_ownership') using LabelEncoder
- Scale features using StandardScaler (all columns except 'defaulted')
- Print a confirmation message: "✅ Real dataset loaded and preprocessed."
- Return the preprocessed DataFrame

### 2. `explore_data(df)`
**Purpose**: Explore the loan amount data.
**Requirements**:
- Implement this function to analyze loan amount distribution
- Display key statistics about the loan amount
- The output must include the phrase "loan amount"
- In the skeleton code, this function raises a NotImplementedError to ensure the test fails until properly implemented

### 3. `sigmoid_demo()`
**Purpose**: Demonstrate the sigmoid activation function.
**Requirements**:
- Calculate the sigmoid of 1.5: sigmoid(1.5) = 1 / (1 + e^(-1.5))
- Print the result: "🧠 Sigmoid(1.5) = [value]"

### 4. `cost_function(y_true, y_pred_prob)`
**Purpose**: Implement a custom log loss cost function.
**Requirements**:
- Add a small epsilon (e.g., 1e-15) to prevent log(0)
- Clip prediction probabilities to avoid extreme values
- Calculate binary cross-entropy: -mean(y_true * log(y_pred_prob) + (1 - y_true) * log(1 - y_pred_prob))
- Return the calculated cost

### 5. `train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl")`
**Purpose**: Train a logistic regression model and evaluate it.
**Requirements**:
- Create a LogisticRegression model with max_iter=1000
- Train the model with the training data
- Save the model to the specified path using joblib.dump()
- Print a confirmation message: "✅ Model trained and saved to '[path]'"
- Make predictions and calculate probabilities
- Calculate the custom cost using the cost_function
- Print evaluation metrics:
  - "🎯 Log Loss (Custom Cost): [value]"
  - "🔍 Sample Predictions: [first 10 predictions]"

## Testing Your Implementation

After implementing the functions according to these requirements, you can run the test file to verify your implementation:

```python
python -m test.test_functional
```

The test file will check if your functions meet the requirements and provide feedback on which tests passed or failed.

## Important Notes

1. Make sure to import all necessary libraries at the beginning of each file.
2. Follow the function signatures exactly as specified in the skeleton code.
3. Ensure that your functions print the expected output messages.
4. The random_state parameter should be set to 42 for reproducibility.
5. Pay attention to the return values of each function, as they are used in subsequent tests.

## Test Behavior with Skeleton Code

With the current skeleton implementation:

- All tests will fail as expected, indicating that proper implementation is required.
- The skeleton code is designed to ensure that tests fail until the functions are properly implemented.
- Each function in the skeleton code either returns None, raises an exception, or lacks the required output to pass the tests.

This behavior is intentional, as it ensures that you need to implement all the required functionality to pass the tests.
