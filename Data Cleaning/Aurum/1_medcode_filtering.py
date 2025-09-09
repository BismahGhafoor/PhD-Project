#import neccessary packages
import pandas as pd

# Path to your "code" document
diabetes_code_doc = "/rfs/LRWE_Proj88/Shared/Codes/aurum_final.txt"

# Read the document in
df_codes = pd.read_csv(diabetes_code_doc, sep='\t', dtype={'medcode': str})
print("df_codes before filtering:")
print(df_codes['type'].value_counts())

# filter out 0 so only type = 1 and 2 remain in the dataframe
df_filtered_codes = df_codes[df_codes['type'].isin([1, 2])].copy()
print("df_codes after filtering:")
print(df_filtered_codes['type'].value_counts())


# Add a 'terminology' column with the value 'medcode'
df_filtered_codes.loc[:, 'terminology'] = 'medcode'

# change medcode header to code
df_filtered_codes.rename(columns={'medcode': 'code'}, inplace=True)
# save the filtered df
df_filtered_codes.to_csv("filtered_diabetes_AURUM_codes.txt", sep="\t", index=False)
