import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('data/house_prices.csv')

# Preprocessing
X = df[['area', 'location', 'bedrooms']]
y = df['price']

# Create preprocessing pipeline
'''preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['location']),
        ('num', 'passthrough', ['area', 'bedrooms'])
    ])
'''

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location']),
        ('num', 'passthrough', ['area', 'bedrooms'])
    ])


# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/price_predictor.pkl')
