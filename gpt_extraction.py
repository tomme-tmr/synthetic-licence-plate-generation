'''
Created on Apr 17 2025  
@author: Tom-Marvin Rathmann

This script extracts license plate numbers from vehicle images (in the context of the project
images with synthetic licence plates) using OpenAI's Vision API.

Therefor it processes all `.png` files/images in the output folder, sends them to the API 
with a fixed prompt, and stores the predictions in a JSON file.

Dependencies:
- openai, pandas, base64, os
- config.py (provides API key and directory paths)

Requires a valid OpenAI Vision API key in `config.py`.
'''


from openai import OpenAI
import os
import base64
import pandas as pd
import config

from config import OpenAI_api_key, DIR

#Set the API key
client = OpenAI(api_key=OpenAI_api_key)
MODEL="gpt-4o-mini"

#Prompt to extract the relevant information
PROMPT = "Return the license plate number of the vehicle. Return just the license plate number without any other text or explanation."

#Path to the image folder
folder_path = DIR['synthetic_pictures']
file_path_gpt_results = DIR['synthetic_pictures'] / f"license_plates_gpt_extraction.json"



#Open the image file
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

#OpenAI Call and extraction of the relevant information
def extractInfoFromImage(base64Image):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64Image}"}
                }
            ]}
        ],
        temperature=0.0,
    )

    lp = response.choices[0].message.content
    return lp


def main():
    results_df = pd.DataFrame(columns=["filename", "license_plate_prediction"])

    #Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            print(filename)
            image_path = os.path.join(folder_path, filename)

            #encode the image
            base64Image = encode_image(image_path)
            
            #extract number plate from the image
            license_plate = extractInfoFromImage(base64Image)
            print(license_plate)

            #add number plate to df
            results_df = results_df._append({"filename": filename, "license_plate_prediction": license_plate}, ignore_index=True)

    #save data frame as json
    results_df.to_json(file_path_gpt_results, orient="records", indent=4)

if __name__ == '__main__':
    main()
