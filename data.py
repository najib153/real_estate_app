import pandas as pd
import numpy as np
import pickle
import os

def load_data(filepath):
    """Load and prepare the real estate dataset"""
    df = pd.read_csv(filepath)
    # Set display options
    pd.set_option('display.max_columns', 50)
    return df

def split_data(df, target_col='price', test_size=0.2, random_state=1234):
    """Split data into training and test sets"""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test



def save_model(model, filename):
    """Save a trained model to file with basic error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def load_model(filename):
    """Load a saved model from file with basic error handling"""
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {filename}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {filename} not found")
        raise
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise