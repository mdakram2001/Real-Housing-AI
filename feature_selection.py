import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
    
# Load the dataset
df = pd.read_csv('my_gurgaon_properties_missing_value_imputation.csv')

# Display the first few rows of the dataset
print(df.head())

# Drop the 'society' and 'price_per_sqft' columns as they are not relevant for now.
train_df = df.drop(columns=['society', 'price_per_sqft'])
print(train_df.head())

"""Plotting the correlation heatmap to visualize the relationships between features."""
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
"""Correlation of features with the target variable 'price'."""
print(train_df.corr()['price'].sort_values(ascending=False)) # Show some strong Multicollinearity between features.
