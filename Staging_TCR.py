import pandas as pd
import numpy as np

#Staging_TCR = pd.read_excel("E:\Staging_TCR.xlsx")
#csv_file_path='E:\Staging.csv'
#Staging = Staging_TCR.to_csv(csv_file_path, index=False)

#print(f"DataFrame saved to {csv_file_path}")


#READ DATASET OF STAGING
Staging = pd.read_csv('E:/Staging.csv')
print(Staging)
print(Staging.info())

#Split into the three tables
dim_TCR = Staging[["pole","N_compte","Montant","date","Agrégat_TCR"]]
dim_Comptes = Staging[["N_compte","Code_compte","Designation_des_comptes","Details_Comptes","N_Comptes2"]]
dim_Production_Immobilise = Staging[["Date_Prod_Immob","Pole_emis","Pole_Reçu","Montant_Prod_Immo"]]

#Editing the dim_TCR dataset
#Clean
#dim_TCR.loc[:,"pole"] = dim_TCR["pole"].str.replace("","Siege")
dim_TCR = dim_TCR.dropna(subset=["N_compte","Montant","date","Agrégat_TCR"])
dim_TCR = dim_TCR.reset_index(drop=True)

#Editing the dim_Comptes dataset
dim_Comptes = dim_Comptes.dropna()
dim_Comptes = dim_Comptes.reset_index(drop=True)

#Editing the dim_Production_Immobilise dataset
dim_Production_Immobilise = dim_Production_Immobilise.dropna()
dim_Production_Immobilise = dim_Production_Immobilise.reset_index(drop=True)


#Save tables to individual csv files
#dim_TCR.to_csv("dim_TCR.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of dim_TCR
plt.figure(figsize=(10, 6))
sns.scatterplot(x='N_compte', y='Montant', data=dim_TCR, hue='Agrégat_TCR')
plt.title('Scatter Plot of Montant vs. N_compte with Hue based on Agrégat_TCR')
plt.show()

# Visualization of dim_Production_Immobilise
plt.figure(figsize=(10, 6))
sns.barplot(x='Pole_emis', y='Montant_Prod_Immo', data=dim_Production_Immobilise)
plt.title('Bar Plot of Montant_Prod_Immo based on Pole_emis')
plt.show()



import statsmodels.api as sm

# Ensure 'N_compte' is numeric in dim_TCR
dim_TCR['N_compte'] = pd.to_numeric(dim_TCR['N_compte'], errors='coerce')  # coerce will turn non-numeric values to NaN

# Regression analysis for dim_TCR
X = sm.add_constant(dim_TCR[['N_compte']])
y = dim_TCR['Montant']

# Drop rows with NaN values in either X or y
data = pd.concat([X, y], axis=1).dropna()

X = data[['const', 'N_compte']]
y = data['Montant']

model = sm.OLS(y, X).fit()
print(model.summary())


import statsmodels.api as sm

# Assuming 'model' is the previously trained OLS model

# Load your new data for prediction
new_data = pd.read_csv('dim_TCR.csv')  # Replace with the path to your new data file

# Ensure 'N_compte' is numeric in the new data
new_data['N_compte'] = pd.to_numeric(new_data['N_compte'], errors='coerce')

# Add a constant to the predictor variable in the new data
new_data_X = sm.add_constant(new_data[['N_compte']])

# Make predictions
predictions = model.predict(new_data_X)

# Add the predictions to the new data
new_data['Predicted_Montant'] = predictions

# Display the new data with predictions
print(new_data[['N_compte', 'Predicted_Montant']])
