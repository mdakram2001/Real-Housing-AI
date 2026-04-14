import numpy as np
import pandas as pd
import re

# display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# read the dataset
df = pd.read_csv('houses.csv')
df.sample(5)

# shape
df.shape

# info
df.info()

# check for duplicates
df.duplicated().sum()

# drop duplicates
df = df.drop_duplicates()

# shape after dropping duplicates
df.shape

# check for missing values
df.isnull().sum()

# Columns to drop -> property_name, link, property_id
df.drop(columns=['link','property_id'], inplace=True)

# rename columns
df.rename(columns={'rate':'price_per_sqft'},inplace=True)
df.head()

# society
df['society'].value_counts()
df['society'].value_counts().shape
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
df['society'].value_counts().shape
df['society'] = df['society'].str.replace('nan','independent')

# price
df['price'].value_counts()
df = df[df['price'] != 'Price on Request']

def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)
        
df['price'] = df['price'].str.split(' ').apply(treat_price)


# price_per_sqft
df['price_per_sqft'].value_counts()
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

# bedrooms
df['bedRoom'].value_counts()
df[df['bedRoom'].isnull()]
df = df[~df['bedRoom'].isnull()]
df.shape
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')


# bathroom
df['bathroom'].value_counts()
df['bathroom'].isnull().sum()
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')


# balcony
df['balcony'].value_counts()
df['balcony'].isnull().sum()
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
df.head()


# additionalRoom
df['additionalRoom'].value_counts()
df['additionalRoom'].fillna('not available',inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()


# floors
df['noOfFloor'].value_counts()
df['noOfFloor'].isnull().sum()
df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
df.head()
df.rename(columns={'noOfFloor':'floorNum'},inplace=True)
df.head()

# facing
df['facing'].value_counts()
df['facing'].fillna('NA',inplace=True)

# area
df['area'] = round((df['price']*10000000)/df['price_per_sqft'])

# property type
df.insert(loc=1,column='property_type',value='house')


df.head()
df.shape
df.info()
df.to_csv('my_house_cleaned.csv',index=False)



