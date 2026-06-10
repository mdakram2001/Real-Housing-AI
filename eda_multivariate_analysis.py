import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_csv('my_gurgaon_properties_cleaned_v2.csv').drop_duplicates()

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

# 4. Property Type vs Bed Room
"""Distribution of the number of bedrooms across different property types. This can help identify which property types typically have more or fewer bedrooms."""
sns.heatmap(pd.crosstab(df['property_type'],df['bedRoom']))
plt.title('Heatmap of Property Type vs Bed Room')
plt.xlabel('Bed Room')
plt.ylabel('Property Type')
plt.show()
# Check For Outliers in Bed Room
print(df[df['bedRoom'] >= 10])
# Pivot Table to show the median price for each combination of property type and number of bedrooms. This can help identify which combinations are generally more expensive or affordable.
plt.figure(figsize=(15,4))
sns.heatmap(pd.pivot_table(df,index='property_type',columns='bedRoom',values='price',aggfunc='mean'),annot=True)

# 5. Property Type vs Floor Number
"""Distribution of the number of floors across different property types. This can help identify which property types are typically found in multi-story buildings."""
sns.barplot(x=df['property_type'],y=df['floorNum'])
plt.title('Median Number of Floors by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Number of Floors')
plt.xticks(rotation=45)
plt.show()
"""Box plot to show the spread of the number of floors for each property type. This can help identify outliers and the range of floors within each category."""
sns.boxplot(x=df['property_type'], y=df['floorNum'])
plt.title('Number of Floors Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Number of Floors')
plt.xticks(rotation=45)
plt.show()
# Check For Outliers in Number of Floors
print(df[(df['property_type'] == 'house') & (df['floorNum'] > 10)])
"""Important Observation: Some houses are located on very high floors, which is unusual. This indicates that there are some houses/villas that are part of multi-story buildings, which is not typical for standalone houses."""

# 6. Property Type vs Age of Property
"""Heatmap of the distribution of the age of property across different property types. This can help identify which property types are generally newer or older."""
sns.heatmap(pd.crosstab(df['property_type'],df['agePossession']))
plt.title('Age of Property Distribution by Property Type')
plt.xlabel('Age of Property')
plt.ylabel('Property Type')
plt.show()
"""Pivot Table to show the median age of property for each property type. This can help identify which property types are generally newer or older."""
sns.heatmap(pd.pivot_table(df,index='property_type',columns='agePossession',values='price',aggfunc='mean'),annot=True)
plt.title('Average Price by Property Type and Age of Property')
plt.xlabel('Age of Property')
plt.ylabel('Property Type')
plt.show()

# 7. Property Type vs Furnishing Status
"""Heatmap of the distribution of furnishing status across different property types. This can help identify which property types are more likely to be furnished, semi-furnished, or unfurnished."""
sns.heatmap(pd.crosstab(df['property_type'],df['furnishing_type']))
plt.title('Furnishing Status Distribution by Property Type')
plt.xlabel('Furnishing Status')
plt.ylabel('Property Type')
plt.show()
"""Pivot Table to show the median price for each combination of property type and furnishing status. This can help identify which combinations are generally more expensive or affordable."""
sns.heatmap(pd.pivot_table(df,index='property_type',columns='furnishing_type',values='price',aggfunc='mean'),annot=True)
plt.title('Average Price by Property Type and Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Property Type')
plt.show()

# 8. Property Type vs Luxury Score
"""Bar plot to show the median luxury score for each property type. This can help identify which property types are generally considered more luxurious."""
sns.barplot(x=df['property_type'],y=df['luxury_score'])
plt.title('Median Luxury Score by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Luxury Score')
plt.xticks(rotation=45)
plt.show()
"""Box plot to show the spread of luxury scores for each property type. This can help identify outliers and the range of luxury scores within each category."""
sns.boxplot(x=df['property_type'], y=df['luxury_score'])
plt.title('Luxury Score Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Luxury Score')
plt.xticks(rotation=45)
plt.show()

# 9. Property Type vs Sector
"""Heatmap of the distribution of sectors across different property types. This can help identify which property types are more common in certain sectors."""
plt.figure(figsize=(15,6))
sns.heatmap(pd.crosstab(df['property_type'],df['sector'].sort_index()))
plt.title('Sector Distribution by Property Type')
plt.xlabel('Sector')
plt.ylabel('Property Type')
plt.show()

# 10. Sector vs Price
"""Average Price Per Sector"""
import re
"""Group by 'sector' and calculate the average price"""
avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()
"""Function to extract sector numbers"""
def extract_sector_number(sector_name):
    match = re.search(r'\d+', sector_name)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number for non-numbered sectors
avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(extract_sector_number)
"""Sort by sector number"""
avg_price_per_sector_sorted_by_sector = avg_price_per_sector.sort_values(by='sector_number')
"""Plot the heatmap"""
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sector_sorted_by_sector.set_index('sector')[['price']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Price per Sector (Sorted by Sector Number)')
plt.xlabel('Average Price')
plt.ylabel('Sector')
plt.show()

# 11. Sector vs Price per Square Foot
"""Average Price Per Square Foot Per Sector"""
avg_price_per_sqft_sector = df.groupby('sector')['price_per_sqft'].mean().reset_index()
avg_price_per_sqft_sector['sector_number'] = avg_price_per_sqft_sector['sector'].apply(extract_sector_number)
"""Sort by sector number"""
avg_price_per_sqft_sector_sorted_by_sector = avg_price_per_sqft_sector.sort_values(by='sector_number')
"""Plot the heatmap"""
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sqft_sector_sorted_by_sector.set_index('sector')[['price_per_sqft']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Price per Square Foot by Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()

# 12. Sector vs Luxury Score
"""Average Luxury Score Per Sector"""
luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()
luxury_score['sector_number'] = luxury_score['sector'].apply(extract_sector_number)
"""Sort by sector number"""
luxury_score_sector = luxury_score.sort_values(by='sector_number')
"""Plot the heatmap"""
plt.figure(figsize=(5, 25))
sns.heatmap(luxury_score_sector.set_index('sector')[['luxury_score']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Luxury Score by Sector (Sorted by Sector Number)')
plt.xlabel('Average Luxury Score')
plt.ylabel('Sector')
plt.show()

# 13. Area vs Price vs Bed Room
"""Scatter plot to show the relationship between area, price, and number of bedrooms. This can help identify if larger properties with more bedrooms tend to be more expensive."""
plt.figure(figsize=(12,8))
sns.scatterplot(x=df[df['area']<10000]['area'],y=df[df['area']<10000]['price'],hue=df['bedRoom'])
plt.title('Area vs Price colored by Bed Room')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# 14. Area vs Price vs Age of Property
"""Scatter plot to show the relationship between area, price, and age of property. This can help identify if newer properties with larger areas tend to be more expensive."""
plt.figure(figsize=(12,8))
sns.scatterplot(x=df[df['area']<10000]['area'],y=df['price'],hue=df['agePossession'])
plt.title('Area vs Price colored by Age of Property')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# 15. Area vs Price vs Furnishing Status
"""Scatter plot to show the relationship between area, price, and furnishing status. This can help identify if furnished properties with larger areas tend to be more expensive."""
plt.figure(figsize=(12,8))
sns.scatterplot(x=df[df['area']<10000]['area'],y=df['price'],hue=df['furnishing_type'].astype('category'))
plt.title('Area vs Price colored by Furnishing Status')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# 16. Bed Room vs Price
"""Bar plot to show the median price for each number of bedrooms. This can help identify if properties with more bedrooms tend to be more expensive."""
sns.barplot(x=df['bedRoom'],y=df['price'],estimator=np.median)
plt.title('Median Price by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Median Price')
plt.show()

# 17. Age of Property vs Price
"""Bar plot to show the median price for each age of property. This can help identify if newer properties tend to be more expensive."""
sns.barplot(x=df['agePossession'],y=df['price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.title('Median Price by Age of Property')
plt.xlabel('Age of Property')
plt.ylabel('Median Price')
plt.show()

# 18. Age of Property vs Area
"""Bar plot to show the median area for each age of property. This can help identify if newer properties tend to have larger areas."""
sns.barplot(x=df['agePossession'],y=df['area'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.title('Median Area by Age of Property')
plt.xlabel('Age of Property')
plt.ylabel('Median Area')
plt.show()

# 19. Furnishing Status vs Price
"""Bar plot to show the median price for each furnishing status. This can help identify if furnished properties tend to be more expensive."""
sns.barplot(x=df['furnishing_type'],y=df['price'],estimator=np.median)
plt.title('Median Price by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Median Price')
plt.show()

# 20. Luxury Score vs Price
"""Scatter plot to show the relationship between luxury score and price. This can help identify if properties with higher luxury scores tend to be more expensive."""
plt.figure(figsize=(12,8))
sns.scatterplot(x=df['luxury_score'],y=df['price'])
plt.title('Luxury Score vs Price')
plt.xlabel('Luxury Score')
plt.ylabel('Price')
plt.show()

# 21. Correlation Heatmap
"""Correlation heatmap to show the relationships between numerical features. This can help identify which features are strongly correlated with price and with each other."""
plt.figure(figsize=(8,8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Correlation of price with other features
print(numeric_df.corr()['price'].sort_values(ascending=False))

# 22. Pair Plot
"""Pair plot to show the relationships between multiple numerical features. This can help identify patterns and correlations between features such as area, price, price per square foot, and luxury score."""
sns.pairplot(df)
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()