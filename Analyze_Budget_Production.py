import pandas as pd
import numpy as np

# Assuming you have a DataFrame named 'budget_production' with your data
# Replace this with your actual DataFrame or load your data from a source

# Pivoting the DataFrame to handle quantities
quantities_df = budget_production.melt(id_vars=['Pole', 'Code_NT', 'act', 'ANNEE', 'CLE_PR'],
                                       var_name='MONT', value_name='Q_propre')

# Extracting month and year from the 'MONT' column
quantities_df['MonthYear'] = quantities_df['ANNEE'] + quantities_df['MONT'] + '01'

# Handling other metrics (values, sales tax, quantities for services, values for services)
metrics = ['V_propre', 'ST', 'Q_Preste', 'V_Preste']

for metric in metrics:
    metric_df = budget_production.melt(id_vars=['Pole', 'Code_NT', 'act', 'ANNEE', 'CLE_PR'],
                                       value_vars=[f'{metric}_1', f'{metric}_2', f'{metric}_3',
                                                   f'{metric}_4', f'{metric}_5', f'{metric}_6',
                                                   f'{metric}_7', f'{metric}_8', f'{metric}_9',
                                                   f'{metric}_10', f'{metric}_11', f'{metric}_12'],
                                       var_name='MONT', value_name=metric)

    metric_df['MonthYear'] = metric_df['ANNEE'] + metric_df['MONT'] + '01'

    # Merge the metric DataFrame with the quantities DataFrame
    quantities_df = pd.merge(quantities_df, metric_df[['Pole', 'Code_NT', 'act', 'MonthYear', metric]],
                             on=['Pole', 'Code_NT', 'act', 'MonthYear'], how='left')

# Reordering columns
result_df = quantities_df[
    ['Pole', 'Code_NT', 'act', 'MonthYear', 'Q_propre', 'V_propre', 'ST', 'Q_Preste', 'V_Preste', 'CLE_PR']]

# Sorting the result
result_df.sort_values(by=['Pole', 'act', 'MonthYear'], inplace=True)

# Display the result
print(result_df)