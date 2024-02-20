# A-Comprehensive-Pandas-Analysis

Make a lot of Analysis , Visualization, Prediction using Python 


This Python code uses Support Vector Machines (SVM) for regression to predict the 'Montant' (amount) based on a dataset. Here's a breakdown of the code:

 # Import Libraries:

pandas: Used for data manipulation and analysis.
train_test_split: From sklearn.model_selection, used to split the dataset into training and testing sets.
SVR (Support Vector Regression): From sklearn.svm, used to perform Support Vector Machine regression.
mean_squared_error: From sklearn.metrics, used to evaluate the performance of the regression model.
Load Dataset:

Reads a CSV file ('STCR_A56.csv') into a pandas DataFrame (df).
Data Preprocessing:

Converts the 'date' column to a datetime object.
Extracts the month and day from the 'date' column.
Select Features and Target Variable:

Defines the feature variable X as the 'N_compte' column and the target variable y as the 'Montant' column.
Train-Test Split:

Splits the dataset into training and testing sets using train_test_split.
SVM Regression:

Creates an SVM regression model with a linear kernel (kernel='linear').
Fits the model on the training data (X_train, y_train).
Evaluate SVM Model:

Predicts the target variable on the test set (X_test) using the trained SVM model.
Calculates the Mean Squared Error (MSE) between the predicted and actual values.
Prediction with SVM:

Creates a new DataFrame (new_data_svm) with a sample value for the 'N_compte' feature.
Uses the trained SVM model to predict the 'Montant' for the new data.
Prints the Mean Squared Error and the predicted 'Montant' value for the new data.
Note: The kernel used in SVM regression is specified as 'linear' in this code, which means a linear kernel is employed for the regression task. The code is designed for predicting 'Montant' based on the 'N_compte' feature using SVM regression.
