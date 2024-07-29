import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('Fish.csv')  # Ensure Fish.csv is in the same folder

# Data preprocessing
X = df.drop(columns=['Weight'])  # Assuming 'Weight' is the target variable
y = df['Weight']

# Encode categorical variables if necessary
X = pd.get_dummies(X, drop_first=True)

# Print columns for debugging
print("Feature columns:", X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
