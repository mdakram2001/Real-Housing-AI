import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()

# 1. Property Type vs Price
"""Distribution of price across different property types. This can help identify which property types are generally more expensive or affordable."""
sns.barplot(x=df['property_type'], y=df['price'], estimator=np.median)
plt.title('Median Price by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Price')
plt.xticks(rotation=45)
plt.show()
"""Box plot to show the spread of prices for each property type. This can help identify outliers and the range of prices within each category."""
sns.boxplot(x=df['property_type'], y=df['price'])
plt.title('Price Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

# 2. Property Type vs Area
"""Distribution of area across different property types. This can help identify which property types generally have larger or smaller areas."""
sns.barplot(x=df['property_type'], y=df['built_up_area'], estimator=np.median)
plt.title('Median Built-Up Area by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Built-Up Area')
plt.xticks(rotation=45)
plt.show()
"""Box plot to show the spread of built-up area for each property type. This can help identify outliers and the range of areas within each category."""
sns.boxplot(x=df['property_type'], y=df['built_up_area'])
plt.title('Built-Up Area Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Built-Up Area')
plt.xticks(rotation=45)
plt.show()
# Removing Extreme Outlier for better visualization
df = df[df['built_up_area'] != 737147]
sns.boxplot(x=df['property_type'], y=df['built_up_area'])
plt.title('Built-Up Area Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Built-Up Area')
plt.xticks(rotation=45)
plt.show()

# 3. Property Type vs Price per Square Foot
"""Distribution of price per square foot across different property types. This can help identify which property types offer better value for money."""
sns.barplot(x=df['property_type'], y=df['price_per_sqft'], estimator=np.median)
plt.title('Median Price per Square Foot by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Price per Square Foot')
plt.xticks(rotation=45)
plt.show()
"""Box plot to show the spread of price per square foot for each property type. This can help identify outliers and the range of price per square foot within each category."""
sns.boxplot(x=df['property_type'], y=df['price_per_sqft'])
plt.title('Price per Square Foot Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price per Square Foot')
plt.xticks(rotation=45)
plt.show()
# Check For Outliers in Price per Square Foot
print(df[df['price_per_sqft'] > 100000][['property_type','society','sector','price','price_per_sqft','area','areaWithType', 'super_built_up_area', 'built_up_area', 'carpet_area']])
