import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("Chennai houseing sale.csv")

# Date parsing
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'], errors='coerce', dayfirst=True)
df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'], errors='coerce', dayfirst=True)
df['BUILD_AGE'] = df['DATE_SALE'].dt.year - df['DATE_BUILD'].dt.year

# Drop missing data
df = df[['MZZONE', 'STREET', 'PARK_FACIL', 'BUILD_AGE', 'N_ROOM', 'SALES_PRICE']].dropna()

# Features and target
X = df.drop(['SALES_PRICE'], axis=1)
y = df['SALES_PRICE']

# Categorical columns
cat_cols = ['MZZONE', 'STREET', 'PARK_FACIL']

# Preprocessing + model
pipeline = Pipeline([
    ('encoder', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, 'simple_house_model.pkl')
print("✅ Simple model saved as simple_house_model.pkl")
