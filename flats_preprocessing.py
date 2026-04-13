import numpy as np
import pandas as pd
import re


# To Print all the columns and rows of the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read the csv file and print 5 random samples of the dataframe
df = pd.read_csv('flats.csv')
df.sample(5)

# Check the shape, info, describe, null values and duplicated values of the dataframe
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum() 

# Drop the columns "link" and "property_id" as they are not useful for our analysis
df.drop(columns=["link", "property_id"], inplace=True)

# Rename the column "area" to "price_per_sqft" as it is more descriptive of the data it contains
df.rename(columns={"area": "price_per_sqft"}, inplace=True)

# Check the unique values of the column "society" and their counts
df['society'].value_counts()
df['society'].value_counts().shape

# The column "society" contains the name of the society where the flat is located. However, it also contains some ratings in the form of "4.5★" or "3★". We will remove these ratings from the society names and convert them to lowercase for better analysis.
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
df['society'].value_counts()

# The column "price" contains the price of the flat in the form of a string. We will remove the commas and convert it to a numeric type for better analysis. We will also remove the rows where the price is "Price on Request" as they do not contain any useful information.
df = df[df['price'] != 'Price on Request']

# The price column contains the price of the flat in the form of a string. We will remove the commas and convert it to a numeric type for better analysis. We will also remove the rows where the price is "Price on Request" as they do not contain any useful information.
def convert_price(price):
    if type(price) == float:
        return price
    else:
        if price[1] == 'Lac':
            return round(float(price[0])/100, 2)
        else:
            return round(float(price[0]), 2)

# The price column contains the price of the flat in the form of a string. We will remove the commas and convert it to a numeric type for better analysis. We will also remove the rows where the price is "Price on Request" as they do not contain any useful information.
df['price'] = df['price'].str.split(' ').apply(convert_price)

# The column "price_per_sqft" contains the price per square foot of the flat in the form of a string. We will remove the commas and convert it to a numeric type for better analysis. We will also remove the rows where the price per square foot is "Price on Request" as they do not contain any useful information.
df['price_per_sqft'].value_counts()
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

# The column "bedRooms" contains the number of bedrooms in the flat in the form of a string. We will extract the numeric part and convert it to a numeric type for better analysis. We will also remove the rows where the number of bedrooms is null as they do not contain any useful information.
df['bedRoom'].value_counts()
df[df['bedRoom'].isnull()]
df = df[~df['bedRoom'].isnull()]
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

# The column "bathRoom" contains the number of bathrooms in the flat in the form of a string. We will extract the numeric part and convert it to a numeric type for better analysis. We will also remove the rows where the number of bathrooms is null as they do not contain any useful information.
df['bathroom'].value_counts()
df['bathroom'].isnull().sum()
df = df[~df['bathroom'].isnull()]
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

# The column "balcony" contains the number of balconies in the flat in the form of a string. We will extract the numeric part and convert it to a numeric type for better analysis. We will also remove the rows where the number of balconies is null as they do not contain any useful information.
df['balcony'].value_counts()
df['balcony'].isnull().sum()
df = df[~df['balcony'].isnull()]
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0') # Balcony column contains '3+', hence we are keeping as string.

# The column "additionalRoom" contains the additional rooms in the flat in the form of a string. We will convert it to lowercase for better analysis. We will also fill the null values with "not available" as they do not contain any useful information.
df['additionalRoom'].value_counts()
df['additionalRoom'].isnull().sum()
df['additionalRoom'].fillna('not available', inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()

# The column "floorNumber" contains the floor number of the flat in the form of a string. We will extract the numeric part and convert it to a numeric type for better analysis. We will also fill the null values with "not available" as they do not contain any useful information.
df['floorNum'].value_counts()
df['floorNum'].isnull().sum()
df[df['floorNum'].isnull()]
df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).str.replace('Ground', '0').str.replace('Basement', '-1').str.replace('Lower', '0').str.extract(r'(\d+)')

# The column "facing" contains the direction the flat is facing in the form of a string. We will convert it to lowercase for better analysis. We will also fill the null values with "NA" as they do not contain any useful information.
df['facing'].value_counts()
df['facing'].isnull().sum()
df['facing'].fillna('NA', inplace=True)

# The column "society" contains the name of the society where the flat is located. We will convert it to lowercase for better analysis. We will also fill the null values with "NA" as they do not contain any useful information.
df['society'].value_counts()
df.insert(loc=4, column='area', value=round((df['price']*10000000)/df['price_per_sqft']))
df.insert(loc=1, column='property_type', value='flat')

# Finally, we will check the info and shape of the dataframe and save it to a new csv file called "my_flats_preprocessed.csv" without the index.
df.info()
df.shape
df.to_csv('my_flats_preprocessed.csv', index=False)


