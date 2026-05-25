import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# This script performs feature engineering on columns 1. areaWithType 2. additionalRooms 3. agePossession 4. furnishDetails 5. features

# Set display options for better visibility of the DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Load the cleaned dataset
df = pd.read_csv('my_gurgaon_properties_cleaned_v1.csv')

print(df.head(1)) # Check the first row of the DataFrame to understand the structure of the data


def get_super_built_up_area(text):

    # Use regular expression to extract the super built-up area value from the text
    # The pattern looks for "Super Built up area" followed by a number (which can be an integer or a decimal)

    match = re.search(r'Super Built up area (\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def get_area(text, area_type):

    # Use regular expression to extract the area value from the text
    # The pattern looks for the area type followed by a number (which can be an integer or a decimal)
    
    match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def convert_to_sqft(text, area_value):

    # Use regular expression to extract the area value in square meters from the text and convert it to square feet
    # The pattern looks for the area value followed by "(xx sq.m.)" where xx is the area in square meters

    if area_value is None:
        return None
    match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
    if match:
        sq_m_value = float(match.group(1))
        return sq_m_value * 10.7639  # Convert square meters to square feet
    return area_value

# Extract super built-up area, built-up area, and carpet area from the 'areaWithType' column and convert them to square feet
df['super_built_up_area'] = df['areaWithType'].apply(get_super_built_up_area)
df['super_built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis=1)

# Extract built-up area and convert it to square feet
df['built_up_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Built up area'))
df['built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

# Extract carpet area and convert it to square feet
df['carpet_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
df['carpet_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)


# Display the relevant columns to verify the extracted and converted area values
df[['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].sample(5)

# Check for duplicates in the DataFrame
df.duplicated().sum()

# Check the shape of the DataFrame after removing duplicates
df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].shape

# Check the rows where all three area columns (super_built_up_area, built_up_area, carpet_area) are null to understand the extent of missing data
df[df['areaWithType'].str.contains('Plot')][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].head(5)

# Check the number of null values in the area columns to understand the extent of missing data
df.isnull().sum()

# Create a DataFrame to analyze the rows where all three area columns are null to understand the characteristics of these rows and decide on the imputation strategy
all_nan_df = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]

# Display the first few rows of the DataFrame where all three area columns are null to analyze the characteristics of these rows
all_nan_df.head()

# Get the indices of the rows where all three area columns are null to facilitate further analysis or imputation
all_nan_index = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].index

# Define a function to extract plot area from the 'areaWithType' column for rows where all three area columns are null, which likely indicates that the property is a plot and not a built-up area
def extract_plot_area(area_with_type):
    match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
    return float(match.group(1)) if match else None

# Apply the function to extract plot area and create a new column 'built_up_area' in the DataFrame for rows where all three area columns are null, as these rows likely represent plots and the plot area can be considered as the built-up area for analysis purposes
all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(extract_plot_area)

# # Display the first few rows of the DataFrame with the extracted plot area values to verify the correctness of the extraction process
all_nan_df.head()

# Update the original DataFrame with the extracted plot area values for rows where all three area columns were null, ensuring that the 'built_up_area' column is populated with the plot area for these rows to facilitate further analysis and modeling
def convert_scale(row):
    if np.isnan(row['area']) or np.isnan(row['built_up_area']):
        return row['built_up_area']
    else:
        if round(row['area']/row['built_up_area']) == 9.0:
            return row['built_up_area'] * 9
        elif round(row['area']/row['built_up_area']) == 11.0:
            return row['built_up_area'] * 10.7
        else:
            return row['built_up_area']

# Apply the function to convert the scale of the 'built_up_area' values based on the ratio of 'area' to 'built_up_area' for rows where all three area columns were null, ensuring that the 'built_up_area' values are adjusted appropriately for further analysis and modeling 
all_nan_df['built_up_area'] = all_nan_df.apply(convert_scale,axis=1)

# Display the first few rows of the DataFrame with the adjusted 'built_up_area' values to verify the correctness of the scaling process
all_nan_df.head()

# Update the original DataFrame with the adjusted 'built_up_area' values for rows where all three area columns were null, ensuring that the 'built_up_area' column is populated with the adjusted plot area for these rows to facilitate further analysis and modeling
df.update(all_nan_df)

# Display the relevant columns of the original DataFrame after updating the 'built_up_area' values to verify that the updates have been applied correctly
df.isnull().sum()



# Now that the area columns have been extracted and cleaned, we can proceed with further feature engineering on the remaining columns such as 'additionalRooms', 'agePossession', 'furnishDetails', and 'features' to create new features that may be useful for modeling.

df['additionalRoom'].value_counts()

# Create new binary columns for each type of additional room based on the 'additionalRoom' column, which contains information about the presence of various types of additional rooms in the property. This will allow us to capture the presence of specific types of additional rooms as features for modeling.
new_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']

# For each type of additional room in the 'additionalRoom' column, create a new binary column in the DataFrame that indicates the presence (1) or absence (0) of that type of additional room for each property. This will allow us to capture the presence of specific types of additional rooms as features for modeling.
for col in new_cols:
    df[col] = df['additionalRoom'].str.contains(col).astype(int)

df.sample(5)[['additionalRoom','study room', 'servant room', 'store room', 'pooja room', 'others']]

print(df.head(5))



#
df['agePossession'].value_counts()

# The 'agePossession' column contains information about the age of the property and its possession status. We can categorize this information into broader categories to create a new feature that captures the age and possession status of the property in a more meaningful way for modeling purposes.
def categorize_age_possession(value):
    if pd.isna(value):
        return "Undefined"
    if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
        return "New Property"
    if "1 to 5 Year Old" in value:
        return "Relatively New"
    if "5 to 10 Year Old" in value:
        return "Moderately Old"
    if "10+ Year Old" in value:
        return "Old Property"
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    try:
        # For entries like 'May 2024'
        int(value.split(" ")[-1])
        return "Under Construction"
    except:
        return "Undefined"

df['agePossession'] = df['agePossession'].apply(categorize_age_possession)

df['agePossession'].value_counts()

df.head()

# The 'furnishDetails' column contains information about the furnishings in the property. We can extract the presence and count of specific furnishings to create new features that capture the furnishing details of the property for modeling purposes.


df.sample(5)[['furnishDetails','features']]

# Extract the unique furnishings from the 'furnishDetails' column to identify the different types of furnishings that are mentioned in the dataset. This will allow us to create new features for each type of furnishing based on their presence and count in the 'furnishDetails' column.
all_furnishings = []
for detail in df['furnishDetails'].dropna():
    furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
    all_furnishings.extend(furnishings)
unique_furnishings = list(set(all_furnishings))

# Define a function to extract the count of a specific furnishing from the 'furnishDetails' column for each row, which will allow us to create new features that capture the presence and count of specific furnishings in the property for modeling purposes.
def get_furnishing_count(details, furnishing):
    if isinstance(details, str):
        if f"No {furnishing}" in details:
            return 0
        pattern = re.compile(f"(\d+) {furnishing}")
        match = pattern.search(details)
        if match:
            return int(match.group(1))
        elif furnishing in details:
            return 1
    return 0

# Create new columns for each unique furnishing based on the 'furnishDetails' column, which contains information about the furnishings in the property. The new columns will capture the presence and count of specific furnishings for modeling purposes. We will first clean the furnishing names to remove any counts and "No" prefixes to create more meaningful
columns_to_include = [re.sub(r'No |\d+', '', furnishing).strip() for furnishing in unique_furnishings]
columns_to_include = list(set(columns_to_include))  # Get unique furnishings
columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]  # Remove empty strings

# Create new columns for each unique furnishing and populate with counts based on the 'furnishDetails' column, which contains information about the furnishings in the property. The new columns will capture the presence and count of specific furnishings for modeling purposes. We will use the 'get_furnishing_count' function to extract the count of each furnishing for each row in the DataFrame.
for furnishing in columns_to_include:
    df[furnishing] = df['furnishDetails'].apply(lambda x: get_furnishing_count(x, furnishing))

# Create a new DataFrame 'furnishings_df' that includes the furnishing details and the newly created columns for each unique furnishing, which will allow us to analyze the furnishing details separately and perform clustering based on these features to categorize the properties based on their furnishings for modeling purposes.
furnishings_df = df[['furnishDetails'] + columns_to_include]

furnishings_df.shape

furnishings_df.drop(columns=['furnishDetails'],inplace=True)

furnishings_df.sample(5)

# Use the Elbow method to determine the optimal number of clusters for the KMeans algorithm based on the furnishing details extracted from the 'furnishDetails' column. We will fit the KMeans model for a range of cluster numbers and calculate the Within-Cluster-Sum-of-Squares (WCSS) to identify the point where adding more clusters does not significantly reduce WCSS, which indicates the optimal number of clusters for categorizing the properties based on their furnishings for modeling purposes. We will first scale the furnishing features using StandardScaler to ensure that all features contribute equally to the clustering process, as the counts of different furnishings may be on different scales and we want to avoid any bias in the clustering results due to differences in feature scales.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(furnishings_df)

wcss_reduced = []

# Use the Elbow method to determine the optimal number of clusters for the KMeans algorithm based on the scaled furnishing features. We will fit the KMeans model for a range of cluster numbers and calculate the Within-Cluster-Sum-of-Squares (WCSS) to identify the point where adding more clusters does not significantly reduce WCSS, which indicates the optimal number of clusters for categorizing the properties based on their furnishings for modeling purposes. We will use the scaled furnishing features to ensure that all features contribute equally to the clustering process, as the counts of different furnishings may be on different scales and we want to avoid any bias in the clustering results due to differences in feature scales.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss_reduced.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(range(1,11), wcss_reduced, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal Number of Clusters (Reduced Range)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Based on the Elbow method plot, we can choose the optimal number of clusters for the KMeans algorithm to categorize the properties based on their furnishings. The optimal number of clusters is typically identified at the point where the WCSS starts to level off, indicating that adding more clusters does not significantly reduce WCSS. In this case, we will choose 3 clusters as it appears to be a reasonable choice based on the plot.
n_clusters = 3

# Fit the KMeans model with the chosen number of clusters based on the Elbow method plot to categorize the properties based on their furnishings. We will use the scaled furnishing features to ensure that all features contribute equally to the clustering process, as the counts of different furnishings may be on different scales and we want to avoid any bias in the clustering results due to differences in feature scales.
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data)

# Predict the cluster assignments for each row
cluster_assignments = kmeans.predict(scaled_data)

# Add the cluster assignments as a new column in the original DataFrame to categorize the properties based on their furnishings for modeling purposes. The new column 'furnishing_type' will indicate the cluster assignment for each property, which can be used as a feature in modeling to capture the furnishing details of the property. We will drop the original furnishing columns to reduce redundancy and simplify the dataset for modeling purposes, as the cluster assignments capture the relevant information about the furnishings in a more structured way for modeling.
df = df.iloc[:,:-18]

df['furnishing_type'] = cluster_assignments

df.sample(5)[['furnishDetails','furnishing_type']]
# 0 -> unfurnished
# 1 -> semifurnished
# 2 -> furnished



# The 'features' column contains information about the various features and amenities available in the property. We can extract the presence of specific features and create new binary features for modeling purposes. We will first convert the string representation of the list of features into an actual list using 'ast.literal_eval' and then use 'MultiLabelBinarizer' to create a binary matrix for each unique feature. This will allow us to capture the presence of specific features as binary features for modeling purposes.
df[['society','features']].sample(5)
df['features'].isnull().sum()
app_df = pd.read_csv('appartments.csv')
app_df.head(2)

# Convert the 'PropertyName' column in the 'app_df' DataFrame to lowercase to facilitate case-insensitive merging with the 'society' column in the original DataFrame, which will allow us to fill the missing values in the 'features' column based on the corresponding entries in the 'app_df' DataFrame. We will merge the two DataFrames on the 'society' and 'PropertyName' columns to fill the missing values in the 'features' column with the 'TopFacilities' information from the 'app_df' DataFrame.

app_df['PropertyName'] = app_df['PropertyName'].str.lower()
temp_df = df[df['features'].isnull()]
temp_df.shape
x = temp_df.merge(app_df,left_on='society',right_on='PropertyName',how='left')['TopFacilities']
df.loc[temp_df.index,'features'] = x.values
df['features'].isnull().sum()

# The 'features' column contains information about the various features and amenities available in the property. We can extract the presence of specific features and create new binary features for modeling purposes. We will first convert the string representation of the list of features into an actual list using 'ast.literal_eval' and then use 'MultiLabelBinarizer' to create a binary matrix for each unique feature. This will allow us to capture the presence of specific features as binary features for modeling purposes.
df['features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])

# Use MultiLabelBinarizer to convert the list of features into a binary matrix, where each column represents a unique feature and the values indicate the presence (1) or absence (0) of that feature for each property. This will allow us to capture the presence of specific features as binary features for modeling purposes.
mlb = MultiLabelBinarizer()
features_binary_matrix = mlb.fit_transform(df['features_list'])

# Convert the binary matrix into a DataFrame
features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

features_binary_df.sample(5)
features_binary_df.shape

wcss_reduced = []

# Use the Elbow method to determine the optimal number of clusters for the KMeans algorithm based on the binary features extracted from the 'features' column. We will fit the KMeans model for a range of cluster numbers and calculate the Within-Cluster-Sum-of-Squares (WCSS) to identify the point where adding more clusters does not significantly reduce WCSS, which indicates the optimal number of clusters for categorizing the properties based on their features.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_binary_df)
    wcss_reduced.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(range(1,11), wcss_reduced, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal Number of Clusters (Reduced Range)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Define the weights for each feature as provided
# Assigning weights based on perceived luxury contribution
# The weights are assigned based on the perceived luxury contribution of each feature, with higher weights indicating features that are generally considered more luxurious or desirable in a property. These weights will be used to calculate a luxury score for each property based on the presence of these features.
weights = {
    '24/7 Power Backup': 8,
    '24/7 Water Supply': 4,
    '24x7 Security': 7,
    'ATM': 4,
    'Aerobics Centre': 6,
    'Airy Rooms': 8,
    'Amphitheatre': 7,
    'Badminton Court': 7,
    'Banquet Hall': 8,
    'Bar/Chill-Out Lounge': 9,
    'Barbecue': 7,
    'Basketball Court': 7,
    'Billiards': 7,
    'Bowling Alley': 8,
    'Business Lounge': 9,
    'CCTV Camera Security': 8,
    'Cafeteria': 6,
    'Car Parking': 6,
    'Card Room': 6,
    'Centrally Air Conditioned': 9,
    'Changing Area': 6,
    "Children's Play Area": 7,
    'Cigar Lounge': 9,
    'Clinic': 5,
    'Club House': 9,
    'Concierge Service': 9,
    'Conference room': 8,
    'Creche/Day care': 7,
    'Cricket Pitch': 7,
    'Doctor on Call': 6,
    'Earthquake Resistant': 5,
    'Entrance Lobby': 7,
    'False Ceiling Lighting': 6,
    'Feng Shui / Vaastu Compliant': 5,
    'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8,
    'Flower Garden': 7,
    'Food Court': 6,
    'Foosball': 5,
    'Football': 7,
    'Fountain': 7,
    'Gated Community': 7,
    'Golf Course': 10,
    'Grocery Shop': 6,
    'Gymnasium': 8,
    'High Ceiling Height': 8,
    'High Speed Elevators': 8,
    'Infinity Pool': 9,
    'Intercom Facility': 7,
    'Internal Street Lights': 6,
    'Internet/wi-fi connectivity': 7,
    'Jacuzzi': 9,
    'Jogging Track': 7,
    'Landscape Garden': 8,
    'Laundry': 6,
    'Lawn Tennis Court': 8,
    'Library': 8,
    'Lounge': 8,
    'Low Density Society': 7,
    'Maintenance Staff': 6,
    'Manicured Garden': 7,
    'Medical Centre': 5,
    'Milk Booth': 4,
    'Mini Theatre': 9,
    'Multipurpose Court': 7,
    'Multipurpose Hall': 7,
    'Natural Light': 8,
    'Natural Pond': 7,
    'Park': 8,
    'Party Lawn': 8,
    'Piped Gas': 7,
    'Pool Table': 7,
    'Power Back up Lift': 8,
    'Private Garden / Terrace': 9,
    'Property Staff': 7,
    'RO System': 7,
    'Rain Water Harvesting': 7,
    'Reading Lounge': 8,
    'Restaurant': 8,
    'Salon': 8,
    'Sauna': 9,
    'Security / Fire Alarm': 9,
    'Security Personnel': 9,
    'Separate entry for servant room': 8,
    'Sewage Treatment Plant': 6,
    'Shopping Centre': 7,
    'Skating Rink': 7,
    'Solar Lighting': 6,
    'Solar Water Heating': 7,
    'Spa': 9,
    'Spacious Interiors': 9,
    'Squash Court': 8,
    'Steam Room': 9,
    'Sun Deck': 8,
    'Swimming Pool': 8,
    'Temple': 5,
    'Theatre': 9,
    'Toddler Pool': 7,
    'Valet Parking': 9,
    'Video Door Security': 9,
    'Visitor Parking': 7,
    'Water Softener Plant': 7,
    'Water Storage': 7,
    'Water purifier': 7,
    'Yoga/Meditation Area': 7
}
# Calculate the luxury score for each property by multiplying the binary features with their corresponding weights and summing them up to create a new feature 'luxury_score' that captures the overall luxury level of the property based on the presence of various amenities and features.
luxury_score = features_binary_df[list(weights.keys())].multiply(list(weights.values())).sum(axis=1)

df['luxury_score'] = luxury_score
df.head()

# After creating the new features based on the 'additionalRoom', 'agePossession', 'furnishDetails', and 'features' columns, we can drop the original columns that were used to create these new features to reduce redundancy and simplify the dataset for modeling purposes. We will drop the 'nearbyLocations', 'furnishDetails', 'features', 'features_list', and 'additionalRoom' columns as they have been transformed into new features that capture the relevant information in a more structured way for modeling.
df.drop(columns=['nearbyLocations','furnishDetails','features','features_list','additionalRoom'],inplace=True)
df.sample(5)

df.to_csv('my_gurgaon_properties_cleaned_v2.csv',index=False)
