import pandas as pd

# Assuming you have a DataFrame named 'Pointage' with columns: code_mat, v_h_sv, v_h_ch, date
# and another DataFrame named 'sous_famille' with columns: code_sfamille, cat

# Replace the following lines with your actual data loading logic
# pointage_df = pd.read_csv('your_pointage_data.csv')
# sous_famille_df = pd.read_csv('your_sous_famille_data.csv')

# Example data loading (replace with your actual data)
pointage_df = pd.DataFrame({
    'code_mat': ['ABCDE1', 'FGHIJ2', 'KLMNO3'],
    'v_h_sv': [10, 15, 20],
    'v_h_ch': [5, 8, 12],
    'date': ['2022-01-01', '2022-01-02', '2022-01-03']
})

sous_famille_df = pd.DataFrame({
    'code_sfamille': ['ABCDE', 'FGHIJ', 'KLMNO'],
    'cat': ['A', 'B', 'C']
})

# Extracting the year from the date column
pointage_df['year'] = pd.to_datetime(pointage_df['date']).dt.year

# Filter data for the year 2022 and where cat is not null
filtered_pointage = pointage_df[(pointage_df['year'] == 2022) & (pointage_df['cat'].notnull())]

# Perform the left join
merged_df = pd.merge(filtered_pointage, sous_famille_df, left_on=pointage_df['code_mat'].str[:5], right_on=sous_famille_df['code_sfamille'], how='left')

# Group by and calculate the sum for the first part of the union
result1 = merged_df.groupby(['cat', 'date']).agg(Montant_global=('v_h_sv', 'sum')).reset_index()

# Create a DataFrame for the second part of the union
result2 = pd.DataFrame({
    'cat': sous_famille_df['cat'],
    'Montant_global': 0,
    'date': pointage_df['date']
})

# Merge with sous_famille for the second part of the union
result2 = pd.merge(result2, sous_famille_df, left_on=result2['code_mat'].str[:5], right_on=sous_famille_df['code_sfamille'], how='left')

# Filter for the year 2022 and where cat is not null
result2 = result2[(result2['year'] == 2022) & (result2['cat'].notnull())]

# Group by for the second part of the union
result2 = result2.groupby(['cat', 'date']).agg(Montant_global=('Montant_global', 'sum')).reset_index()

# Concatenate the results and sort by cat and date
final_result = pd.concat([result1, result2]).sort_values(['cat', 'date']).reset_index(drop=True)

print(final_result)
