import streamlit as st
import pandas as pd
import pickle
from data import load_data
from modeling import train_random_forest
from visualization import plot_decision_tree

# Configure page
st.set_page_config(page_title="Real Estate Predictor", layout="wide")

# Title
st.title("🏠 Real Estate Price Prediction")
st.write("Predict property prices using machine learning")

# Sidebar inputs
with st.sidebar:
    st.header("Property Details")
    year_sold = st.number_input("Year Sold", 2000, 2025, 2022)
    property_tax = st.number_input("Property Tax", 0, 10000, 2000)
    insurance = st.number_input("Insurance", 0, 500, 100)
    beds = st.number_input("Bedrooms", 1, 10, 2)
    baths = st.number_input("Bathrooms", 1, 10, 2)
    sqft = st.number_input("Square Feet", 500, 10000, 1500)
    year_built = st.number_input("Year Built", 1900, 2025, 2000)
    lot_size = st.number_input("Lot Size", 0, 100000, 5000)
    basement = st.selectbox("Basement", [0, 1], format_func=lambda x: "Yes" if x else "No")
    popular = st.selectbox("Popular Home", [0, 1], format_func=lambda x: "Yes" if x else "No")
    recession = st.selectbox("Recession Period", [0, 1], format_func=lambda x: "Yes" if x else "No")
    property_age = st.number_input("Property Age", 0, 100, 10)
    property_type = st.radio("Property Type", ["House", "Bunglow", "Condo"], index=0)

# Convert property type to binary columns
property_type_map = {
    "House": [0, 0],
    "Bunglow": [1, 0], 
    "Condo": [0, 1]
}
bunglow, condo = property_type_map[property_type]

# Load model
@st.cache_resource
def load_saved_model():
    try:
        with open('model/RE_Model', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_saved_model()

# Prediction button
if st.button("Predict Price"):
    input_data = {
        'year_sold': year_sold,
        'property_tax': property_tax,
        'insurance': insurance,
        'beds': beds,
        'baths': baths,
        'sqft': sqft,
        'year_built': year_built,
        'lot_size': lot_size,
        'basement': basement,
        'Popular_Home': popular,
        'Recession_Period': recession,
        'Age': property_age,
        'Bunglow': bunglow,
        'Condo': condo
    }
    
    if model:
        try:
            input_df = pd.DataFrame([input_data])[model.feature_names_in_]
            prediction = model.predict(input_df)[0]
            st.success(f"### Predicted Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Data exploration section
if st.checkbox("Show sample data"):
    df = load_data("data/final_real_estate.csv")
    st.dataframe(df.head())