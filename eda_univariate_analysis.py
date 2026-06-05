import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('my_gurgaon_properties_cleaned_v2.csv')

# Getting an overview of the dataset, using head, info, describe, shape and checking for duplicates
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.duplicated().sum()) 

# Removing duplicates if any
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())



# Univariate analysis for features
# 1. Property Type
df['property_type'].value_counts().plot(kind='bar')
plt.title('Distribution of Property Types')
plt.xlabel('Property Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show() 
""" 
Observations: 
1. Flats are approx 75% while houses are around 25%
2. No missing values
"""



# 2. Society
print(df['society'].value_counts().shape)
print(df['society'].value_counts())
df[df['society'] != 'independent']['society'].value_counts(normalize=True).cumsum().head(75)
society_counts = df['society'].value_counts()
# Frequency distribution for societies
frequency_bins = {
    "Very Hign (>100)":(society_counts>100).sum(),
    "High (50-100)":((society_counts>=50) & (society_counts<=100)).sum(),
    "Average (10-49)":((society_counts>=10) & (society_counts<=40)).sum(),
    "Low (2-9)":((society_counts>=2) & (society_counts<=9)).sum(),
    "Very Low (1)":(society_counts==1).sum(),
}
print(frequency_bins)
# Top 10 societies
df[df['society'] != 'independent']['society'].value_counts().head(10).plot(kind='bar')
plt.show()
df['society'].isnull().sum()
"""
Observations:
1. Around 13% properties comes under independent tag.
2. There are 675 societies.
3. The top 75 societies have 50% of the properties and the rest 50% of the properties comes under the remaining 600 societies.
4. 1 missing value
"""


# 3. Sector
print(df['sector'].value_counts().shape)
df['sector'].value_counts().head(10).plot(kind='bar')
plt.show()
sector_counts = df['sector'].value_counts()
sector_frequency_bins = {
    "Very Hign (>100)":(sector_counts>100).sum(),
    "High (50-100)":((sector_counts>=50) & (sector_counts<=100)).sum(),
    "Average (10-49)":((sector_counts>=10) & (sector_counts<=40)).sum(),
    "Low (2-9)":((sector_counts>=2) & (sector_counts<=9)).sum(),
    "Very Low (1)":(sector_counts==1).sum(),
}
print(sector_frequency_bins)
print(df['sector'].isnull().sum())
"""
Observations:
1. Total 115 unique sectors in the dataset.
2. Frequency Distribution:
    Very Hign (>100): 3, 
    High (50-100): 25, 
    Average (10-49): 52, 
    Low (2-9): 23), 
    Very Low (1): 1
3. No missing values
"""



# 4. Price
print(df['price'].isnull().sum())
print(df['price'].describe())
sns.histplot(df['price'], kde=True, bins=50)
plt.show()
sns.boxplot(x=df['price'], color='lightgreen')
plt.grid()
plt.show()
"""
Observation:
Descriptive Statistics->
count    3660
mean     2.53 Cr
std      2.98
median   1.52 Cr
min      0.07 Cr
max      31.5 Cr

1. Histogram shows that most properties are priced in the lower range i.e below 5 Cr while some are beyond 10 Cr (Data is right skewed)
2. Boxplot shows the spread of the data and potential outliers. Properties above 10 Cr might be considered as outliers.
3. There are 17 missing values in the price column.
"""

# 5. Skewness & Kurtosis (Price)
skewness = df['price'].skew()
kurtosis = df['price'].kurt()
print(skewness,kurtosis)
"""
Observation:
Skewness = 3.28 ---> positive skew/right skew.
Kurtosis = 14.93 ---> K>3 indicates a distribution with heavier tails & more outliers compare to normal distribution.
"""

# 6. Quantile Analysis (Price)
quantiles = df['price'].quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
print(quantiles)
"""------------------------------------------"""
Q1 = df['price'].describe()['25%']
Q3 = df['price'].describe()['75%']
IQR = Q3 - Q1
print(IQR)
"""------------------------------------------"""
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(lower_bound, upper_bound)
"""------------------------------------------"""
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(outliers.shape)
print(outliers['price'].describe())
"""------------------price binning------------------------"""
bins = [0, 1, 2, 3, 5, 10, 20, 50]
labels = ['0-1 Cr', '1-2 Cr', '2-3 Cr', '3-5 Cr', '5-10 Cr', '10-20 Cr', '20-50 Cr']
pd.cut(df['price'], bins=bins, labels=labels, right=False).value_counts().sort_index().plot(kind='bar')
plt.show()
"""---------------------ecdf plot-------------------------"""
ecdf = df['price'].value_counts().sort_index().cumsum()/len(df['price'])
plt.plot(ecdf.index, ecdf, marker='*', linestyle='none')
plt.xlabel('Price (Cr)')
plt.title('ECDF of Property Prices')
plt.grid()
plt.show()
"""
Observations:
1. 1% of the properties are priced below 0.25 Cr, 25% are below 0.95 Cr, 50% are below 1.52 Cr, 75% are below 2.75 Cr, 95% are below 8.50 Cr and 99% are below 15.264 Cr, indicating very few properties are priced above 15 Cr.
2. The interquartile range (IQR) is 1.8 Cr, which indicates that the middle 50% of the properties are priced between 0.95 Cr and 2.75 Cr.
3. The lower bound for outliers is -1.75 Cr (which is not possible in real estate) and the upper bound is 5.45 Cr, indicating that properties priced above 5.45 Cr can be considered as outliers.
4. There are 425 outliers in the dataset, with a mean price of 9.23 Cr, a median price of 8.0 Cr, and a maximum price of 31.5 Cr.
"""

# Henc, it is rightskewed distribution, to convert it in normal distribution we can apply log transformation.
"""Distribution plot without log transformation"""
plt.subplot(1, 2, 1)
sns.histplot(df['price'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Property Prices (Original)')
plt.xlabel('Price (Cr)')
plt.ylabel('Frequency')
"""Distribution plot with log transformation"""
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), kde=True, bins=50, color='salmon')
plt.title('Distribution of Property Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
"""
Observations:
np.log1p(x) : It computes the natural logarithm of (1 + x), which is useful for handling data very close to zero and negative values in the data.
We can use np.expm1(x) to reverse the log transformation, which computes e^x - 1.
"""
skewness_log = np.log1p(df['price']).skew()
kurtosis_log = np.log1p(df['price']).kurt()
print(skewness_log, kurtosis_log)
plt.figure(figsize=(15, 6))
"""Distribution plot without log transformation"""
plt.subplot(1, 2, 1)
sns.boxplot(x=df['price'], color='lightgreen')
plt.title('Boxplot of Property Prices (Original)')
plt.xlabel('Price (Cr)')
plt.ylabel('Frequency')
plt.grid()
"""Distribution plot with log transformation"""
plt.subplot(1, 2, 2)
sns.boxplot(x=np.log1p(df['price']), color='salmon')
plt.title('Boxplot of Property Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# 7. Price Per Sqft
print(df['price_per_sqft'].isnull().sum())
print(df['price_per_sqft'].describe())
"""Histogram and Boxplot for Price Per Sqft"""
sns.histplot(df['price_per_sqft'], kde=True, bins=50, color='lightblue')
plt.title('Distribution of Price Per Sqft')
plt.xlabel('Price Per Sqft')
plt.ylabel('Frequency')
plt.grid()
plt.show()
"""Boxplot for Price Per Sqft"""
sns.boxplot(x=df['price_per_sqft'], color='lightgreen')
plt.title('Boxplot of Price Per Sqft')
plt.xlabel('Price Per Sqft')
plt.ylabel('Frequency')
plt.grid()
plt.show()
"""IQR for Price Per Sqft"""
iqr_price_per_sqft = df['price_per_sqft'].quantile(0.75) - df['price_per_sqft'].quantile(0.25)
print(f"IQR for Price Per Sqft: {iqr_price_per_sqft}")
"""
Observations:
The price per sqft has 17 missing values, with a mean of 13892, a median of 9020, a minimum of 4 and a maximum of 600000.
The boxplot indicates the presence of outliers in the price per sqft, with some values significantly higher/lower than the rest.
The IQR is relatively compact, but there are many data points beyound the "whiskers" of the boxplot, indicating a wide range of price per sqft values in the dataset.
"""

# 8. Bedroom
print(df['bedRoom'].isnull().sum())
print(df['bedRoom'].describe())
"""Distribution of Bedrooms"""
df['bedRoom'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.grid()
plt.show()
"""Pie chart for Bedrooms"""
df['bedRoom'].value_counts(normalize=True).head().plot(kind='pie', autopct='%0.2f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'lightpink'])
plt.title('Distribution of Bedrooms')
plt.ylabel('Frequency')
plt.show()
"""
Observations:
The bedroom column has 0 missing values, with a mean of 3.3 bedrooms, a median of 3, a minimum of 1 and a maximum of 21. Most properties have 3 bedrooms (approx 40%), followed by 2 bedrooms (approx 26%%), and 4 bedrooms (approx 20%). Properties with 5 or more bedrooms are less common, making up around 10% of the dataset.
"""

# 9. Bathroom
print(df['bathroom'].isnull().sum())
print(df['bathroom'].describe())
"""Distribution of Bathrooms"""
df['bathroom'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.grid()
plt.show()
"""Pie chart for Bathrooms"""
df['bathroom'].value_counts(normalize=True).head().plot(kind='pie', autopct='%0.2f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'lightpink'])
plt.title('Distribution of Bathrooms')
plt.ylabel('Frequency')
plt.show()
"""
Observations:
The bathroom column has 0 missing values, with a mean of 3.4 bathrooms, a median of 3, a minimum of 1 and a maximum of 10. Most properties have 3 bathrooms (approx 32%), followed by 2 bathrooms (approx 31%), and 4 bathrooms (approx 25%). Properties with 5 or more bathrooms are less common, making up around 13% of the dataset.
"""

# 10. Balcony
print(df['balcony'].isnull().sum())
print(df['balcony'].describe())
"""Distribution of Balconies"""
df['balcony'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Balconies')
plt.xlabel('Number of Balconies')
plt.ylabel('Frequency')
plt.show()
"""Pie chart for Balconies"""
df['balcony'].value_counts(normalize=True).plot(kind='pie', autopct='%0.2f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'lightpink'])
plt.title('Distribution of Balconies')
plt.ylabel('Frequency')
plt.show()
"""
Observations:
The balcony column has 0 missing values, it is a categorical column having 5 unique categories. Most properties have 3+ balcony (approx 32%), followed by 3 balconies (approx 30%), and 2 balconies (approx 25%). Properties with 1 or no balcony are less common, making up around 13% of the dataset.
"""

# 11. Floor Number of the Property
print(df['floorNum'].isnull().sum())
print(df['floorNum'].describe())
"""Distribution of Floor Number"""
df['floorNum'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Floor Number')
plt.xlabel('Floor Number')
plt.ylabel('Frequency')
plt.show()
"""Box Plot for Floor Number"""
sns.boxplot(x=df['floorNum'], color='lightgreen')
plt.title('Boxplot of Floor Number')
plt.xlabel('Floor Number')
plt.ylabel('Frequency')
plt.show()
"""
Observations:
1. The majority of properties are located between the ground floor and the 25th floor.
2. Floors 1-4 are the most common, with the 3rd floor being the most frequently occurring floor number.
3. There are some properties located on higher floors, but their frequency is significantly lower, indicating that they are less common in the dataset.
4. The box plot shows that majority of the properties are located on lower floors.
5. The IQR lies between approx 2nd & 4th floors.
6. Data points beyond the whiskers (above 21st floor) can be considered as outliers, indicating that properties located on higher floors are less common in the dataset. 
"""

# 12. Facing
print(df['facing'].isnull().sum())
print(df['facing'].fillna('NA', inplace=True))
print(df['facing'].describe())
