import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load('models/price_predictor.pkl')

# App title
st.title('🏠 House Price Prediction')
st.write('Predict house prices based on area, location, and bedroom count')

# Sidebar
st.sidebar.header('Input Features')

# Input form
with st.form('prediction_form'):
    area = st.slider('Area (sq ft)', 500, 5000, 1500)
    location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])
    bedrooms = st.slider('Bedrooms', 1, 6, 2)
    
    submitted = st.form_submit_button('Predict Price')

# Prediction
if submitted:
    input_data = pd.DataFrame([[area, location, bedrooms]], 
                            columns=['area', 'location', 'bedrooms'])
    prediction = model.predict(input_data)[0]
    
    st.success(f'Predicted Price: ${prediction:,.2f}')

# Data visualization
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