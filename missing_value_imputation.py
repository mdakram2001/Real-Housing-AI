import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('my_gurgaon_properties_outlier_treatment.csv')

# Check for missing values
print(df.isnull().sum())

# Missing value imputation
# 1. For Built-up Area
sns.scatterplot(x=df['built_up_area'], y=df['super_built_up_area']) # Strong Linear Relationship
plt.title('Built-up Area vs Super Built-up Area')
plt.xlabel('Built-up Area')
plt.ylabel('Super Built-up Area')
plt.show()
sns.scatterplot(x=df['built_up_area'], y=df['carpet_area']) # Strong Linear Relationship
plt.title('Built-up Area vs Carpet Area')
plt.xlabel('Built-up Area')
plt.ylabel('Carpet Area')
plt.show()

"""Find all the row that does not have built_up_area, super_built_up_area and carpet_area."""
(df['built_up_area'].isnull()) & (df['super_built_up_area'].isnull()) & (df['carpet_area'].isnull()) # No row has all three columns missing

"""Find all the row that contain all three columns."""
all_present_df = df[~((df['built_up_area'].isnull()) | (df['super_built_up_area'].isnull()) | (df['carpet_area'].isnull()))] # Rows with all three columns filled
# Calculate the ratios
super_to_built_up_ratio = (all_present_df['super_built_up_area'] / all_present_df['built_up_area']).median() # Average Super Built-up to Built-up Ratio
carpet_to_built_up_ratio = (all_present_df['carpet_area'] / all_present_df['built_up_area']).median() # Average Carpet to Built-up Ratio
print(f"Median Super Built-up to Built-up Ratio: {super_to_built_up_ratio}")
print(f"Median Carpet to Built-up Ratio: {carpet_to_built_up_ratio}")

"""Find all the row that does not have built_up_area but has super_built_up_area and carpet_area."""
sbc_df = df[(df['built_up_area'].isnull()) & ~(df['super_built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
"""Impute the missing built_up_area using the average of the two ratios calculated above."""
sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area'] / super_to_built_up_ratio) + (sbc_df['carpet_area'] / carpet_to_built_up_ratio)) / 2), inplace=True)
df.update(sbc_df) # Update the original dataframe with the imputed values
print(df.isnull().sum()) # Check for missing values after imputation

"""Find all the row that contains super_built_up_area but does not have built_up_area and carpet_area."""
sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
"""Impute the missing built_up_area using the ratio of super_built_up_area to built_up_area calculated above."""
sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area'] / super_to_built_up_ratio), inplace=True)
df.update(sb_df) # Update the original dataframe with the imputed values
print(df.isnull().sum()) # Check for missing values after imputation

"""Find all the row that contains carpet_area but does not have built_up_area and super_built_up_area."""
cb_df = df[~(df['carpet_area'].isnull()) & (df['built_up_area'].isnull()) & (df['super_built_up_area'].isnull())]
"""Impute the missing built_up_area using the ratio of carpet_area to built_up_area calculated above."""
cb_df['built_up_area'].fillna(round(cb_df['carpet_area'] / carpet_to_built_up_ratio), inplace=True)
df.update(cb_df) # Update the original dataframe with the imputed values
print(df.isnull().sum()) # Check for missing values after imputation

"""Scatter plot to visualize the relationship between built_up_area and price after imputation."""
sns.scatterplot(x=df['built_up_area'], y=df['price'])
plt.title('Built-up Area vs Price after Imputation')
plt.xlabel('Built-up Area')
plt.ylabel('Price')
plt.show()

"""This scatter plot shows the relationship between built_up_area and price but there are some anomalies in the data, which itself is forming linear relationship."""
anomoly_df = df[(df['built_up_area'] > 2000) & (df['price'] > 2.5)][['built_up_area', 'price', 'area']]
anomoly_df.sample(5) 
anomoly_df['built_up_area'] = anomoly_df['area']
df.update(anomoly_df) # Update the original dataframe with the anomalies
sns.scatterplot(x=df['built_up_area'], y=df['price'])
plt.title('Built-up Area vs Price after Imputation')
plt.xlabel('Built-up Area')
plt.ylabel('Price')
plt.show()

"""Removing Columns that are not required for further analysis."""
df.drop(columns=['area', 'super_built_up_area', 'carpet_area', 'area_room_ratio', 'areaWithType'], inplace=True)


# 2. For Floor Number
print(df['floor_number'].isnull().sum()) # Check for missing values in floor_number
sns.countplot(df['floor_number']) # Count plot to visualize the distribution of floor_number
plt.title('Distribution of Floor Numbers')
plt.xlabel('Floor Number')
plt.ylabel('Count')
plt.show()

med_floor_num = df[df['property_type'] == 'house']['floorNum'].median() # Median floor number for houses
df['floor_number'].fillna(med_floor_num, inplace=True) # Impute missing floor numbers with the median for houses
print(df['floor_number'].isnull().sum()) # Check for missing values after imputation


# 3. Facing
print(df['facing'].isnull().sum()) # A lot of missing values in facing column
df['facing'].value_counts().plot(kind='pie', autopct='%0.2f%%')
df.drop(columns=['facing'], inplace=True) # Drop the facing column due to high number of missing values
print(df.isnull().sum()) # Final check for missing values after all imputation and dropping columns

# 4. Soceity
print(df['society'].isnull().sum()) # Check for missing values in society column
df.drop(index=2536, inplace=True) # Drop the row with missing society value
print(df.isnull().sum()) # Final check for missing values after dropping the row with missing society value


# 5. Age Possession having value undefined
print(df[df['agePossession'] == 'Undefined']) # Check for missing values in age_possession column

def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']
df['agePossession'] = df.apply(mode_based_imputation, axis=1) # Impute missing age possession values based on mode of similar properties
print(df['agePossession'].value_counts()) # Check the distribution of age possession values after imputation

def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']
df['agePossession'] = df.apply(mode_based_imputation2, axis=1) # Impute remaining missing age possession values based on mode of properties in the same sector
print(df['agePossession'].value_counts()) # Check the distribution of age possession values after second round of imputation

def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']
df['agePossession'] = df.apply(mode_based_imputation3, axis=1) # Impute remaining missing age possession values based on mode of properties of the same type
print(df['agePossession'].value_counts()) # Check the distribution of age possession values after third round of imputation

print(df.isnull().sum()) # Final check for missing values after all imputation steps

df.to_csv('my_gurgaon_properties_missing_value_imputation.csv', index=False) # Save the cleaned dataset to a new CSV file
