import pickle
import pandas as pd


# Postal_Short = input("Enter the Postal short (First three letters): ")
# Style = input("Enter the Style of the house that you are looking for: ")
# Type = input("Enter Type of the house: ")
# Cluster = input("Enter the cluster: ")
# List_Price = int(input("Enter the List price of the house: "))
# Cluster_Price = float(input("Enter the Cluster Price: "))
# Taxes = int(input("Enter the Taxes of house: "))
# Cluster_Tax = float(input("Enter the cluster tax of the house: "))
# Bedrooms = int(input("Enter the no. of bedrooms: "))
# Washrooms = int(input("Enter the no. of washrooms: "))
# Basement1 = input("Enter the Basement1: ")
# Days_on_market = int(input("Enter days on the market: "))
# Exterior1 = input("Enter the exterior1: ")
# Garage_Type = input("Entet the Garage type: ")
# lat = float(input("Enter the latitude of the house: "))
# lng = float(input("Entet the longitude of the house: "))

Washrooms = "l9z"
Bedrooms = "bungalow"
Maintenance = "Detached"
Taxes = "l9z Detached"
# List_Price = 699900
Cluster_Price = 816858.798639456
Taxes = 2841.93
Cluster_Tax = 3618.19810884354
Bedrooms = 2
Washrooms = 2
Basement1 = "Finished"
Days_on_market = 200
Exterior1 = "Brick"
Garage_Type = "Other"
lat = 44.4738343
lng = -80.0682258


path = "/Users/thrilok/Desktop/mantra_collab_job/work_files/latest_broko_code"


data = {
    "Postal_Short": [Postal_Short],
    "Style": [Style],
    "Type": [Type],
    "Cluster": [Cluster],
    "Cluster_Price": [Cluster_Price],
    "Taxes": [Taxes],
    "Cluster_Tax": [Cluster_Tax],
    "Bedrooms": [Bedrooms],
    "Washrooms": [Washrooms],
    "Basement1": [Basement1],
    "Days_On_Market": [Days_on_market],
    "Exterior1": [Exterior1],
    "Garage_Type": [Garage_Type],
    "lat": [lat],
    "lng": [lng],
}

data = pd.DataFrame(data)
print(data)

# # Convert categorical features to categorical data type
categorical_features = [
    column for column, dtype in data.dtypes.items() if dtype == object
]
print(categorical_features)
data[categorical_features] = data[categorical_features].astype("category")

# To Load the model
pickled_model = pickle.load(open(path + "/models//conda.pkl", "rb"))

# Make a prediction
predicted_price = pickled_model.predict(data[:1])


print(f"Predicted Price: {predicted_price[0]}")
