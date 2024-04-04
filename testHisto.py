import pandas as pd
#chemin_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/data_missFrance.csv'
#data = pd.read_csv(chemin_ana)
#print(data.columns)

#data_subset = data[['annee', 'cheveux']]
#print(data_subset)

import pandas as pd

chemin_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/data_missFrance.csv'

# Read the CSV file
data = pd.read_csv(chemin_ana,delimiter=';')
print(data.columns)

# Check if the DataFrame is empty
if data.empty:
    print("DataFrame is empty!")
else:
    # Check if the columns exist in the DataFrame
    if 'annee' in data.columns and 'cheveux' in data.columns:
        # Subset the DataFrame
        data_subset = data[['annee', 'cheveux']]
        print(data_subset.head())
    else:
        print("Columns 'annee' and 'cheveux' do not exist in the DataFrame.")
