import numpy as np
import pandas as pd

# Loading the cleaned dataframes for flats and houses
flats = pd.read_csv('my_flats_preprocessed.csv')
houses = pd.read_csv('my_house_cleaned.csv')

# Merging the two dataframes, ignoring the index to avoid duplicate indices in the merged dataframe
df = pd.concat([flats,houses],ignore_index=True)

# Shuffling the rows of the dataframe, again ignoring the index to avoid any issues with duplicate indices after shuffling
df = df.sample(df.shape[0],ignore_index=True)

# Printing the shapes of the individual dataframes and the merged dataframe to verify the merge and shuffle operations
print(flats.shape)
print(houses.shape)
print(df.shape)

df.to_csv('my_gurgaon_properties.csv',index=False)