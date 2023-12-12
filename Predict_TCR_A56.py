import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('D:/Save/Dataset/STCR_A56.csv')

# Data Preprocessing
# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
# Extract relevant features from the date column
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select features (X) and target variable (y)
X = df[['N_compte']].copy()
y = df['Montant']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction
# Assume you have a new data point for prediction (replace it with your actual data)
new_data = pd.DataFrame({'N_compte': [75]})
prediction = model.predict(new_data)
print(f'Predicted Montant: {prediction[0]}')

# Compare Original vs Predicted values
comparison = pd.DataFrame({'Original Montant': y_test, 'Predicted Montant': y_pred})
print(comparison.head(10))  # Displaying the first 10 rows for comparison


