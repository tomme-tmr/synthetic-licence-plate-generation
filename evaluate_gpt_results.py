"""
Created on Apr 17 2025  
@author: Tom-Marvin Rathmann

This script evaluates the accuracy of license plate predictions from GPT by comparing them 
to the given ground truth labels from the synthetic image metadata.

It loads prediction results and reference labels from JSON files, cleans and merges the data, 
and calculates evaluation metrics including:
- Accuracy
- Character Error Rate (CER)

Dependencies:
- pandas, json, re
- scikit-learn (for accuracy_score)
- jiwer (for CER)
- config.py (for path definitions)
"""


import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from jiwer import cer
from config import DIR

#creates a dataframe based on the json file in the given folder
def json_file_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    df = pd.DataFrame(json_data)

    return df

#clean the predict column
def clean_license_plate(plate):
    if pd.isnull(plate) or len(plate) > 14:
        return pd.NA

    #Removing spaces and special characters, keeping only alphanumeric characters
    return re.sub(r'[^A-Za-z0-9]', '', plate.strip())


#defining relevant file paths
file_path_holy_list = DIR['synthetic_pictures'] / f"holy_list.json"
file_path_gpt_results = DIR['synthetic_pictures'] / f"license_plates_gpt_extraction.json"

#creating dataframes based on the jason files
df_holy_list = json_file_to_dataframe(file_path_holy_list)
df_gpt_results = json_file_to_dataframe(file_path_gpt_results)

#creating a new column in the dataframe, joing the list elements into one string
df_holy_list["np_label"] = df_holy_list["np_identity"].apply(lambda x: ''.join(x))


#Merge der DataFrames based on the columns manual_name und filename
df_holy_list = df_holy_list.merge(
    df_gpt_results[['filename', 'license_plate_prediction']],
    left_on='manual_name',
    right_on='filename',
    how='left'
)

df_holy_list.rename(columns={'license_plate_prediction': 'np_predicted_label'}, inplace=True)
df_holy_list.drop(columns=['filename'], inplace=True)


#applying the cleaning function
df_holy_list['np_predicted_label'] = df_holy_list['np_predicted_label'].apply(clean_license_plate)



#drop empty rows
df_holy_list_striped = df_holy_list.dropna()

#define label and prediction
y_true = df_holy_list_striped['np_label']
y_pred = df_holy_list_striped['np_predicted_label']
y_true_list = y_true.to_list()
y_pred_list = y_pred.to_list()


print(df_holy_list.head(10))

#calculating the metrics
num_columns = df_holy_list.shape[0]

#Number of calls, where no prediction can be made
no_prediction_count = df_holy_list['np_predicted_label'].isna().sum()

#number of exact matches
matches = y_true == y_pred
num_matches = matches.sum()

#calculates accuracy
accuracy = accuracy_score(y_true, y_pred)

#Charracter Error Rate
CER = cer(y_true_list, y_pred_list)

#print metrics
print("Anzahl Fahrzeugbilder:", num_columns)
print("Anzahl nicht möglicher Vorhersagen:", no_prediction_count)
print("Anzahl korrekter Vorhersagen:", num_matches)
print("Accuracy:", accuracy)

print("CER:", CER)