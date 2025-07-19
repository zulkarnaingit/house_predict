import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
import sklearn

# --- VERSION COMPATIBILITY FIX ---
# Patch for missing attribute in older sklearn versions
if not hasattr(ColumnTransformer, '_name_to_fitted_passthrough'):
    ColumnTransformer._name_to_fitted_passthrough = {}

# --- MODEL LOADING ---
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    try:
        model = joblib.load('models/price_predictor.pkl')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# --- APP LAYOUT ---
st.title('🏠 House Price Prediction')
st.write('Predict house prices based on area, location, and bedroom count')

# Load model
model = load_model()

if model is not None:
    # --- INPUT FORM ---
    with st.form('prediction_form'):
        area = st.slider('Area (sq ft)', 500, 5000, 1500)
        location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])
        bedrooms = st.slider('Bedrooms', 1, 6, 2)
        submitted = st.form_submit_button('Predict Price')

    # --- PREDICTION ---
    if submitted:
        try:
            # Prepare input data (ensure correct column order)
            input_data = pd.DataFrame([[area, location, bedrooms]], 
                                    columns=['area', 'location', 'bedrooms'])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted Price: ${prediction:,.2f}')
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Common issues:")
            st.write("- Input data format doesn't match training data")
            st.write("- Model expects different feature names/order")

# --- DATA VISUALIZATION ---
try:
    st.header('Price Distribution')
    df = pd.read_csv('data/house_prices.csv')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='bedrooms', y='price', data=df, ax=ax[0])
    ax[0].set_title('Price by Bedroom Count')

    sns.boxplot(x='location', y='price', data=df, ax=ax[1])
    ax[1].set_title('Price by Location')

    st.pyplot(fig)

    # Data table
    st.header('Sample Data')
    st.dataframe(df.head(10))
    
except Exception as e:
    st.warning(f"Couldn't load visualization data: {str(e)}")

# --- VERSION INFO ---
st.sidebar.markdown("### Environment Info")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
