import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_csv('my_gurgaon_properties_cleaned_v2.csv').drop_duplicates()

print("Initial shape of the dataset:", df.shape)
print("Columns in the dataset:", df.columns.tolist())

# Outlier Treatment
# 1. Price Outliers
sns.displot(df['price'])
plt.title('Distribution of Price')
plt.show() # It is not a normal distribution, it is right skewed.
sns.boxplot(x=df['price'])
plt.title('Boxplot of Price')
plt.show()
"""Calculate the IQR for price"""
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
"""Defining the lower and upper bounds for outliers"""
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
"""Identify outliers in price"""
outliers_price = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
num_outliers_price = outliers_price.shape[0]
print("Number of price outliers:", num_outliers_price)
stats_outliers_price = outliers_price['price'].describe()
print("Statistics of price outliers:", stats_outliers_price)
"""
Observations:
On the basis of price column, we can say that there are some genuine outliers in the dataset but there are some data errors as well.
"""

# 2. Price per Square Foot Outliers
sns.displot(df['price_per_sqft'])
plt.title('Distribution of Price per Square Foot')
plt.show() # It is not a normal distribution, it is right skewed.
sns.boxplot(x=df['price_per_sqft'])
plt.title('Boxplot of Price per Square Foot')
plt.show()
"""Calculate the IQR for price per square foot"""
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1
"""Defining the lower and upper bounds for outliers"""
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
"""Identify outliers in price per square foot"""
outliers_price_per_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]
num_outliers_price_per_sqft = outliers_price_per_sqft.shape[0]
print("Number of price per square foot outliers:", num_outliers_price_per_sqft)
stats_outliers_price_per_sqft = outliers_price_per_sqft['price_per_sqft'].describe()
print("Statistics of price per square foot outliers:", stats_outliers_price_per_sqft)
"""Reviewing the outliers in price per square foot, we can see that there are some data errors in the area column which is leading to very high price per square foot values. We will correct those data errors and then recalculate the price per square foot for those outliers."""
outliers_price_per_sqft['area'] = outliers_price_per_sqft['area'].apply(lambda x:x*9 if x<1000 else x)
outliers_price_per_sqft['price_per_sqft'] = round((outliers_price_per_sqft['price']*10000000)/outliers_price_per_sqft['area'])
print(outliers_price_per_sqft['price_per_sqft'].describe())
df.update(outliers_price_per_sqft)
sns.displot(df['price_per_sqft'])
plt.title('Distribution of Price per Square Foot after Outlier Treatment')
plt.show()
sns.boxplot(x=df['price_per_sqft'])
plt.title('Boxplot of Price per Square Foot after Outlier Treatment')
plt.show()
"""Now again identifying the price per square foot outliers which are greater than 50000 and removing them because their  count is very less and they are not genuine outliers but data errors."""
outliers_price_per_sqft = df[df['price_per_sqft'] > 50000]
num_outliers_price_per_sqft = outliers_price_per_sqft.shape[0]
print("Number of price per square foot outliers after treatment:", num_outliers_price_per_sqft)
df = df[df['price_per_sqft'] <= 50000]
sns.displot(df['price_per_sqft'])
plt.title('Distribution of Price per Square Foot after Removing Extreme Outliers')
plt.show()

# 3. Area Outliers
print("Statistics of area column:", df['area'].describe())
sns.displot(df['area'])
plt.title('Distribution of Area')
plt.show() # It is not a normal distribution, it is right skewed.
sns.boxplot(x=df['area'])
plt.title('Boxplot of Area')
plt.show()
print("Rows with area > 100000:", df[df['area']>100000])
"""Reviewing the rows with area greater than 100000, we can see that there are some data errors in the area column which is leading to very high area values. Hence we will remove those rows because they are not genuine outliers but data errors."""
df = df[df['area'] <= 100000]
sns.displot(df['area'])
plt.title('Distribution of Area after Removing Extreme Outliers')
plt.show()
sns.boxplot(x=df['area'])
plt.title('Boxplot of Area after Removing Extreme Outliers')
plt.show()
"""After removing the rows with area greater than 100000, we are checking if they are genuine outliers or data errors."""
print(df[df['area'] > 10000].sort_values('area',ascending=False))
df.drop(index=[818, 1796, 1123, 2, 115, 3649, 2503, 1471], inplace=True)
print(df[df['area'] > 10000].sort_values('area',ascending=False))
"""We can see that there are some data errors in the area column which is leading to very high area values. Hence we will correct those data errors and then recalculate the area for those outliers."""
df.loc[48,'area'] = 115*9
df.loc[300,'area'] = 7250
df.loc[2666,'area'] = 5800
df.loc[1358,'area'] = 2660
df.loc[3195,'area'] = 2850
df.loc[2131,'area'] = 1812
df.loc[3088,'area'] = 2160
df.loc[3444,'area'] = 1175
"""After correcting the data errors in area column, we can see that there are some genuine outliers in the area column but there are no data errors. Hence we will keep those genuine outliers because they are not affecting our analysis and they are not data errors."""
sns.displot(df['area'])
plt.title('Distribution of Area after Correcting Data Errors')
plt.show()
sns.boxplot(x=df['area'])
plt.title('Boxplot of Area after Correcting Data Errors')
plt.show()
df['area'].describe()

# 4. Bedroom"""
sns.displot(df['bedRoom'])
plt.title('Distribution of Bedroom')
plt.show()
sns.boxplot(x=df['bedRoom'])
plt.title('Boxplot of Bedroom')
plt.show()
print(df['bedRoom'].describe())
print(df[df['bedRoom'] > 10].sort_values('bedRoom',ascending=False))
"""Reviewing the rows with bedroom greater than 10, we can see that there are some data errors in the bedroom column which is leading to very high bedroom values. Hence we will remove those rows because they are not genuine outliers but data errors."""
df = df[df['bedRoom'] <= 10]
sns.displot(df['bedRoom'])
plt.title('Distribution of Bedroom after Removing Extreme Outliers')
plt.show()
sns.boxplot(x=df['bedRoom'])
plt.title('Boxplot of Bedroom after Removing Extreme Outliers')
plt.show()
print(df['bedRoom'].describe())

# 4. Bathroom
sns.displot(df['bathroom'])
plt.title('Distribution of Bathroom')
plt.show()
sns.boxplot(x=df['bathroom'])
plt.title('Boxplot of Bathroom')
plt.show()
print(df['bathroom'].describe())
"""Reviewing the rows with bathroom greater than 10, lookslike those are genuine outliers because there are some properties with very high bathroom values. Hence we will keep those genuine outliers because they are not affecting our analysis and they are not data errors."""
print(df[df['bathroom'] > 10].sort_values('bathroom',ascending=False))

# 5. Super Built-up Area
sns.displot(df['super_built_up_area'])
plt.title('Distribution of Super Built-up Area')
plt.show()
sns.boxplot(x=df['super_built_up_area'])
plt.title('Boxplot of Super Built-up Area')
plt.show()
print(df['super_built_up_area'].describe())
"""Reviewing the rows with super built-up area greater than 6000, There are some large properties with very high super built-up area values. Hence we will keep those genuine outliers because they are not affecting our analysis and they are not data errors."""
print(df[df['super_built_up_area'] > 6000])

# 6. Built-up Area
sns.displot(df['built_up_area'])
plt.title('Distribution of Built-up Area')
plt.show()
sns.boxplot(x=df['built_up_area'])
plt.title('Boxplot of Built-up Area')
plt.show()
print(df['built_up_area'].describe())
"""Reviewing the rows with built-up area greater than 10000"""
print(df[df['built_up_area'] > 10000])

# 7. Carpet Area
sns.displot(df['carpet_area'])
plt.title('Distribution of Carpet Area')
plt.show()
sns.boxplot(x=df['carpet_area'])
plt.title('Boxplot of Carpet Area')
plt.show()
print(df[df['carpet_area'] > 10000])
df.loc[2131,'carpet_area'] = 1812 # Because the area column value for that row is 1812. Hence we will correct the data error in carpet area column for that row.
print(df[df['carpet_area'] > 10000])

# 8. Luxury Score
sns.displot(df['luxury_score'])
plt.title('Distribution of Luxury Score')
plt.show()
sns.boxplot(x=df['luxury_score'])
plt.title('Boxplot of Luxury Score')
plt.show()

# 9. Additionals Checking
"""After treating the outliers in price and area columns, we are checking the price per square foot column again to see if there are any outliers in that column because there were some data errors in area column which was leading to very high price per square foot values. Hence we will recalculate the price per square foot for all the rows in the dataset after treating the outliers in price and area columns."""
df['price_per_sqft'] = round((df['price']*10000000)/df['area'])
sns.displot(df['price_per_sqft'])
plt.title('Distribution of Price per Square Foot')
plt.show()
sns.boxplot(x=df['price_per_sqft'])
plt.title('Boxplot of Price per Square Foot')
plt.show()
print(df[df['price_per_sqft'] > 42000]) # Few Properties but they looklike genuine data points. Some others looks like outliers/data errors if we consider their bivariate relationship with area and bedroom columns. Hence we will check the area to bedroom ratio for those rows to identify the data errors in area column.
x = df[df['price_per_sqft'] <= 20000] # Considering <=20000, because they are mostlt genuine data points.
print((x['area']/x['bedRoom']).quantile(0.05)) # Finding the 5th percentile of area to bedroom ratio for the rows with price per square foot less than or equal to 20000, because they are mostly genuine data points. Ratio less than 250 looks can be considered as outliers/data errors.
print(df[(df['area']/df['bedRoom'])<250]) # There are a some rows with area to bedroom ratio less than 250. Hence we will check those rows to identify the data errors/ inconsistencies in area/bedroom column.
"""Distribution of Bedroom vs. Built-up Area"""
sns.lmplot(x='bedRoom', y='area', data=df)
plt.title('Scatter Plot of Bedroom vs. Built-up Area')
plt.show()
df['area_room_ratio'] = df['area']/df['bedRoom']
(df[df['area_room_ratio']<250])['bedRoom'].value_counts() # Counting the number of rows with area to bedroom ratio less than 250
df = df[df['area_room_ratio'] >= 100] # Removing the rows with area to bedroom ratio less than 100 because they are not genuine data points but data errors. One can't very less area for many bedrooms.
"""After removing the rows with area to bedroom ratio less than 100, we are checking the rows with area to bedroom ratio less than 250 to identify the data errors in area or bedroom column. We will check the distribution of bedroom column for those rows and if there are some rows with very high bedroom values, then we will correct those bedroom values because they are leading to very low area to bedroom ratio values which are not genuine data points but data errors."""
outliers_df = df[(df['area_room_ratio'] < 250 ) & (df["bedRoom"] > 3)]
outliers_df['bedRoom'] = round(outliers_df['bedRoom']/outliers_df['floorNum'])
df.update(outliers_df)
df['area_room_ratio'] = df['area']/df['bedRoom']
print(df[(df['area_room_ratio'] < 250 ) & (df["bedRoom"] > 4)].shape) # Some properties are still there with area to bedroom ratio less than 250 because there floorNum is 1 resulting in same values as before. (approx 13 rows)
df = df[~((df['area_room_ratio'] < 250 ) & (df["bedRoom"] > 4))] # Removing those rows because they are not genuine data points but data errors. One can't very less area for many bedrooms.
sns.lmplot(x='bedRoom', y='area', data=df)
plt.title('Scatter Plot of Bedroom vs. Built-up Area after Removing Data Errors')
plt.show()
df.to_csv('my_gurgaon_properties_outlier_treatment.csv', index=False)
