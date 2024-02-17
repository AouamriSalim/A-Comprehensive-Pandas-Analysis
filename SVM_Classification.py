#SVM Classification to predict Montant
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('D:/Save/Dataset/STCR_A56.csv')

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select features (X) and target variable (y)
X = df[['N_compte']]
y = df['Montant']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Regression
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate SVM Model
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f'SVM - Mean Squared Error: {mse_svm}')

# Prediction with SVM (for example)
new_data_svm = pd.DataFrame({'N_compte': [75]})
prediction_svm = svm_model.predict(new_data_svm)
print(f'SVM - Predicted Montant: {prediction_svm[0]}')
'''