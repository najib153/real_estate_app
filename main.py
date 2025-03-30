# Import modules
from data import load_data, split_data, save_model, load_model
from modeling import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model
from visualization import plot_decision_tree, show_data
import logging
import pandas as pd


# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='real_estate.log'
   
)

def load_data(filepath):
    """Load data from CSV file with error handling"""
 
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise

def main():
    """Main function to run the real estate prediction"""
    try:
        logging.info("Starting real estate price prediction")
        
        # 1. Load data
        df = load_data("final_real_estate.csv")
        print("\nFirst 5 rows of data:")
        print(df.head())
        
        # [Rest of your existing code here]
        
        logging.info("Prediction completed successfully")
        
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    # 1. Load and prepare data
    filepath = r"data\final_real_estate.csv"
    df = load_data(filepath)

    show_data(df)

    print("\n[2/4] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    show_data(X_train)
    print("\nTraining target samples:")
    show_data(y_train)
    show_data(X_test)
    print("\nTest target samples:")
    show_data(y_test)

    # 2. Train and evaluate models
    # Linear Regression
    lr_model, lr_train_mae = train_linear_regression(X_train, y_train)
    lr_test_mae = evaluate_model(lr_model, X_test, y_test)

    # Decision Tree
    dt_model, dt_train_mae = train_decision_tree(X_train, y_train)
    dt_test_mae = evaluate_model(dt_model, X_test, y_test)

    # Random Forest
    rf_model, rf_train_mae = train_random_forest(X_train, y_train)
    rf_test_mae = evaluate_model(rf_model, X_test, y_test)

    # 3. Visualize Decision Tree
    plot_decision_tree(dt_model, feature_names=dt_model.feature_names_in_)

    # 4. Save and load model
    save_model(rf_model, 'model/RE_Model')
    loaded_model = load_model('model/RE_Model')

    # Example prediction
    import pandas as pd

    # Example prediction with proper feature names
    sample_data = {
       'year_sold': [2012],
       'property_tax': [216],
       'insurance': [74],
       'beds': [1],
       'baths': [1],
       'sqft': [618],
       'year_built': [2000],
       'lot_size': [600],
       'basement': [1],
       'Popular_Home': [0],
       'Recession_Period': [0],
       'Age': [12],
       'Bunglow': [0],
       'Condo': [0]
       }

    # Convert to DataFrame with same column order as training
    sample_df = pd.DataFrame(sample_data, columns=rf_model.feature_names_in_)
    prediction = rf_model.predict(sample_df)
    print(f"Predicted price: ${prediction[0]:,.2f}")
































