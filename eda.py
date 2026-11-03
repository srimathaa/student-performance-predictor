
import pandas as pd

# Load the real dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Display dataset information
print("DATASET SHAPE:")
print(data.shape)

print("\nCOLUMN NAMES:")
print(data.columns.tolist())

print("\nFIRST 5 RECORDS:")
print(data.head())

print("\nDATASET INFORMATION:")
data.info()

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nSTATISTICAL SUMMARY:")
print(data.describe())
