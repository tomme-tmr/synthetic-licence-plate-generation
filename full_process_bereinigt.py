"""
Created on Thu Nov 24 21:00:17 2023
Last Update: 17 Apr 2025
@author: LarsG, updated by Tom-Marvin Rathmann


This script is used for the synthetic modification of real vehicle images by adding 
artificial numberplates and (optional) environmental effects such as rain, dirt, fog, or different light settings.

Its main goal is to enhance datasets for machine learning models or computer vision pipelines, 
especially for license plate detection and recognition under diverse and challenging conditions.

Key functionalities:
- Insertion of synthetic numberplates (1-line or 2-line) using perspective and geometric transformations
- Optional addition of synthetic weather artifacts such as:
    * Fog, rain, snow, and sun reflections
    * Dirt or shadows on the numberplate
    * Brightness adjustments
    * Simmulation of car headlights
- Generation of multiple variations per original image (based on config file)
- Storage of all metadata and transformation settings in a JSON log file

The system is fully configurable via `config.yaml` and makes use of a modular architecture 
for rendering, effect application, and numberplate synthesis.

Required modules:
    - Rwa_NumberPlate_function_all.py
    - Random_Digits.py
    - Image_Rendering.py
    - Recognition_Tuv_Sticker.py
    - add_different_weather_conditions.py
    - config.py (for number of licence plates and filepaths)

Note:
- All generated images and metadata will be saved in the output directory, defined in the config.py.
- Currently, the creation of licence plates from motorcycles is not fully supported.

Update Information (April, Projektarbeit - Tom-Marvin Rathmann):
- weather effect integration

Update Information (April, Masterarbeit - Tom-Marvin Rathmann):
- file path handling
- config file integration
- integration of the effect configuration
- weather effect optimization (depth estimation and licence plate background colouring)
"""

from  Rwa_NumberPlate_function_all import generate_number_plate
from Image_Rendering import apply_fog, apply_dust, apply_darkness
from add_different_weather_conditions import add_plate_specific_effect, add_weather_effect, apply_darkness, set_effect_styles, generate_depth_map
from config import DIR

import json
import numpy as np
from PIL import Image, ImageDraw
import math
import cv2
import yaml


#Define the path where the JSON file is located.
file_path = DIR['fahrzeugbilder'] / f"labels_COCO.json"

#Read the config file for image effects and number of pictures
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

#Number of images with synthetic number plate will be set based on the predefined number in the config file. Default value is 1
j = config.get("number_of_variations", 1)


#Load JSON-file which includes the labeling data of the car image templates. 
with open(file_path, 'r') as file:
    data = json.load(file)
liste_erstellt = False
image_counter = False

#Iteration through image and annotation data
for annotation in data['annotations']:
    image_id = annotation['image_id']
    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    bbox = annotation['bbox']
    xy_ratio = bbox[2]/bbox[3]
    
    category_info = next((img for img in data["annotations"] if img["image_id"] == image_id), None)
    car_category = category_info["category_id"]-1
    type_of_np = data['categories'][car_category]["name"]      

    #Distinguish which type of raw numberplate is required.
    #To activate the creation of motorbike numberplate, uncomment the code below. (currently not fully supported)
    if xy_ratio < 2 and "motorbike" not in type_of_np:
        width_raw_NP = 340
        height_raw_NP = 200
        typ = "no_motorbike"
    elif xy_ratio > 2:
        width_raw_NP = 521
        height_raw_NP = 111
    elif xy_ratio < 2 and "motorbike" in type_of_np:
        continue
        #width_raw_NP = 300
        #height_raw_NP = 200
        #typ = "motorbike"
    



    image_id = annotation['image_id']
    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        
    category_info = next((img for img in data["annotations"] if img["image_id"] == image_id), None)
    car_category = category_info["category_id"]-1
    type_of_np = data['categories'][car_category]["name"]
    if "rear" in type_of_np:
        view = "rear"
    elif "front" in type_of_np:
        view = "front"
  

    if image_info:
        image_file_name = image_info['file_name']

        #Define the path of the car image template folder
        image_path = str(DIR["fahrzeugbilder"] / f"{image_file_name}")

        #Load specific car image template according to the labels and type of numberplate
        original_img = Image.open(image_path)
        original_width, original_height = original_img.size

        #Update by Tom-Marvin Rathmann
        #generate a depth map for the picture to apply different effect strength based on the picture depth
        depth_map = generate_depth_map(image_path)

        #Mathmatics calculations to get the geometry of the numperplate on the car image template. 
        y1 = annotation ['segmentation'][0][1]
        y2 = annotation ['segmentation'][0][3]
        y3 = annotation ['segmentation'][0][5]
        y4 = annotation ['segmentation'][0][7]
        x1 = annotation ['segmentation'][0][0]
        x2 = annotation ['segmentation'][0][2]
        x3 = annotation ['segmentation'][0][4]
        x4 = annotation ['segmentation'][0][6]
        
        #Calculation of the length of the numberplate which is mounted on car image template.
        #Calculation is done by using the Pythagorean theorem.
        length_numberplate = math.sqrt((y2-y1)**2+(x2-x1)**2)
        height_numberplate = math.sqrt((y4-y1)**2+(x4-x1)**2)
       

##############----------------------------------------#######################
        #Geometric calculations for left-hand rotation
        if (y1-y2)<8:
            
            rotation_angle_z = math.atan2((x1-x4),(y4-y1))
            rotation_angle_y3 = math.atan2((y2-y1),(x2-x1))
            rotation_angle_phi = math.atan2((y4-y3),(x3-x4))              
            rotation_angle_eta = math.atan2((x2-x3),(y2-y3))
            
            
            x4_new = height_raw_NP * math.tan(rotation_angle_z)
            x4_new = -height_raw_NP * math.sin(rotation_angle_z)
    
            x2_new = width_raw_NP * math.sin(math.pi/2-rotation_angle_z)
            x2_new = width_raw_NP * math.cos(rotation_angle_y3)
            x3_new = width_raw_NP * math.cos(rotation_angle_z)+x4_new
            x3_new = x2_new - height_raw_NP * math.sin(rotation_angle_z)
            x3_new = x2_new - height_raw_NP * math.sin(rotation_angle_eta)     
           
            y2_new = width_raw_NP * math.sin(rotation_angle_y3)
            y3_new = height_raw_NP - math.tan(math.atan2((y4-y3),(x3-x4)))*width_raw_NP
            y3_new = height_raw_NP + width_raw_NP * math.tan(rotation_angle_y3)
            y4_new = height_raw_NP * math.cos(rotation_angle_z)
            
        else:        
            #Geometric calculation for right-hand rotation if y2 is higher than y1. 
            #Respectively y2 is 15 pixels higher than y1.
            #x2 | y2 is the reference point
            #calculation of the angles for trigonometric functions
            rotation_angle_alpha = math.atan2((y1-y2),(x2-x1))
            rotation_angle_beta = math.atan2((x4-x1),(y4-y1))
            rotation_angle_gamma = math.atan2((x3-x2),(y3-y2))             
            #Calculation of new x-coordinates of the cornerstones
            x1_new = width_raw_NP-(width_raw_NP * math.cos(rotation_angle_alpha))
            x3_new = width_raw_NP+(height_raw_NP * math.sin(rotation_angle_gamma))
            x4_new = (x1_new-(height_raw_NP*math.sin(rotation_angle_beta)))
            #Calculation of new y-coordinates of the cornerstones
            y1_new = width_raw_NP * math.sin(rotation_angle_alpha)
            y3_new = height_raw_NP * math.cos(rotation_angle_gamma)
            y4_new = y1_new + (height_raw_NP*(math.cos(rotation_angle_beta)))
            
################################################
        #Depending on how many numberplates are wanted for one specific car image template, 
        #the for loop is executed. 
        for i in range(j):
        #Decision if 2-line-numberplate or single line numberplate is needed.
        #Based on the ratio of the proportions of the numberplate.
            img = Image.open(image_path)
            if xy_ratio < 2:
                line_NP = 2
                #Type of "car" for 2-line numberplates already indicated in the code above.
            else:
                line_NP = 1
                typ = "car"
            
            #Call function to generate a synthetic numberplate
            img_buffer = generate_number_plate(line_NP,view,typ, img, [x1,x2,x3,x4], [y1,y2,y3,y4])



            ########## Update Tom-Marvin Rathmann - Start (I)
            #Module integration (add_different_weather_conditions)
            #plate sepific effect integration
            
            numberplate_with_effect = img_buffer[0]
            effect_setting = config.get("effect_setting", 'random')
            
            #When the user decides the effect should be random (in config file)
            if effect_setting == 'random':
                effects = set_effect_styles()
            
            #When the user sets a specified effect in the config file 
            elif effect_setting == 'specified':
                for effect in config.get("effects", []):
                    if effect.get("type") == "light_setting":
                        light_setting = effect.get("mode")
                        light_effect_strength = effect.get("strength")
                    elif effect.get("type") == "plate_effect":
                        plate_effect = effect.get("mode")
                        plate_effect_strength = effect.get("strength")
                    elif effect.get("type") == "weather_effect":
                        weather_effect = effect.get("mode")
                        weather_effect_strength = effect.get("strength")
                    elif effect.get("type") == "headlights":
                        headlights_mode = effect.get("mode")

                    
                effects = {
                    "light": {
                        "effect": light_setting,
                        "strength": light_effect_strength
                    },
                    "plate": {
                        "effect": plate_effect,
                        "strength": plate_effect_strength
                    },
                    "weather": {
                        "effect": weather_effect,
                        "strength": weather_effect_strength
                    },
                    "headlights": {
                        "mode": headlights_mode
                    }
                }

            #For all other selections (not 'randome' and not 'specified') no effect will be added
            else:
                effects = {
                    "light": {
                        "effect": 'bright',
                        "strength": None
                    },
                    "plate": {
                        "effect": [],
                        "strength": None
                    },
                    "weather": {
                        "effect": [],
                        "strength": None
                    },
                    "headlights": {
                        "mode": 'off'
                    }
                }

            #plate specific effect will be adde
            numberplate_with_effect = add_plate_specific_effect(numberplate_with_effect, effect_setting, effects)
            ########## Tom-Marvin Rathmann - End (I)



            #Caching of the generate synthetic numberplate. 
            numberplate_with_effect.save(str(DIR["input"] / f"numberplate_buffer.png")) #RGBA-Format
            
            
            #Load the synthetic numberplate from the cache again.
            img_dummy1 = cv2.imread(str(DIR["input"] / "numberplate_buffer.png"), cv2.IMREAD_UNCHANGED) #RGBA-Format



            ############ GEOMETRIC / PERSPECTIVE  TRANSFORMATION ######################
            if (y1-y2)<8:
                #Perspective transformation for numberplates with y1 higher than y2 - called "standard".
                #Get the height and the width of the synthetic numberplate.
                height_np, width_np = img_dummy1.shape[:2]
                #Define the cornstones of the source (synthetic numberplate).
                src = np.array([[0, 0], [width_np, 0], [width_np, height_np], [0, height_np]], dtype=np.float32)
                #Define the cornerstones of the transformed numberplate (based on the previous calculations).
                dst = np.array([[0, 0], [x2_new, y2_new], [x3_new, y3_new], [x4_new,y4_new]], dtype=np.float32)
            else:
                #Perspective transformation for numberplates with y2 higher than y1.
                #Get the height and the width of the synthetic numberplate.
                height_np, width_np = img_dummy1.shape[:2]
                #Define the cornstones of the source (synthetic numberplate).
                src = np.array([[0, 0], [width_np, 0], [width_np, height_np], [0, height_np]], dtype=np.float32)
                #Define the cornerstones of the transformed numberplate (based on the previous calculations).
                dst = np.array([[x1_new, y1_new], [width_np, 0], [x3_new, y3_new], [x4_new,y4_new]], dtype=np.float32)
                
            #Geometric calculations of the destination geometry (perspective transformed numberplate)   
            #The values are needed to creat a transparent mask
            #Otherwise the initiall counterlines of the synthetic numberplate are still visible
            max_length_dst_index = np.argmax(dst[:, 0])
            min_length_dst_index = np.argmin(dst[:, 0])
            max_length_dst_value = int(dst[max_length_dst_index,0]) 
            min_length_dst_value = int(dst[min_length_dst_index,0]) 
            
            max_height_dst_index = np.argmax(dst[:, 1])
            min_height_dst_index = np.argmin(dst[:, 1])
            max_height_dst_value = int(dst[max_height_dst_index,1])
            min_height_dst_value = int(dst[min_height_dst_index,1])
            
            
            M = cv2.getPerspectiveTransform(src, dst)
            
            dsize = (max_length_dst_value - min_length_dst_value, max_height_dst_value - min_height_dst_value)  # Size of the mask
            
            # Erstelle eine leere Ausgangsmaske mit einem transparenten Rand
            mask = np.zeros((img_dummy1.shape[0], img_dummy1.shape[1], 4), dtype=np.uint8)
            mask[:, :, 3] = 0  # Setze den Alpha-Kanal auf vollständig transparent
                                
            
            #Apply the perspective transformation to the mask to mark the area of the transformed image
            transformed_mask = cv2.warpPerspective(mask, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255,0 ))
        
            #Apply the perspective transformation to the input image
            transformed_img = cv2.warpPerspective(img_dummy1, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            
            #Paste the transformed image into the mask to obtain the transparent border
            final_img = cv2.addWeighted(transformed_img, 1, transformed_mask, 1, 0)
            
            #Save the image with the transparent border
            cv2.imwrite(str(DIR["input"] / f"transformed_image2.png"), final_img)
            
            
##############################################################################

            #geometric ratio calculation 
                        
            width, height = dsize
            resize_final_img = cv2.resize(final_img, (int((length_numberplate/width_raw_NP) * width), int((height_numberplate/height_raw_NP) * height)))
            
            cv2.imwrite(str(DIR["input"] / f"transformed_image3.png"), resize_final_img)
            
            resize_final_img1 = Image.open(str(DIR["input"] / f"transformed_image3.png"))
                  
            
            #Calculate the position to place the rectangular image inside the bounding box
            min_x = min(x1, x2, x3, x4)
            min_y = min(y1, y2, y3, y4)
            max_x = max(x1, x2, x3, x4)
            max_y = max(y1, y2, y3, y4)
            center_x = (min_x + max_x) / 2 - (resize_final_img1.width / 2)
            center_y = (min_y + max_y) / 2 - (resize_final_img1.height / 2)
                
            #Insert the rectangular image once at the marked points of the bounding box
            img.paste(resize_final_img1, (int(center_x), int(center_y)), resize_final_img1)
            
            img.info['metadata'] = img_buffer[1]
            
            #Save new image with the metadata
            neuer_dateiname = "bild_mit_metadaten.jpg"
            manual_name = f"{image_id}_{i}.png"




            ########## Tom-Marvin Rathmann - Start (II)
            #weather effect for the whole image will be added
            img = add_weather_effect(img, depth_map, effect_setting, effects)

            #if the lighting on the image should be changed, the code will be executed and the darkness effect will be added
            if effects['light']['effect'] == 'dark':
                np_width, np_height = resize_final_img1.size
                np_center_x = center_x + np_width/2
                np_center_y = center_y + np_height/2
                darkness_strength = effects['light']['strength']
                headlights_mode = effects['headlights']['mode']

                img = apply_darkness(img, depth_map, np_center_x, np_center_y, np_width, np_height, darkness_strength, headlights_mode)
            ########## Tom-Marvin Rathmann - End (II)

            file_path = str(DIR["synthetic_pictures"] / f"{manual_name}")
            img.save(file_path)

            
            
    ##########################
            ############ SAVE METADATA TO JSON-FILE ######################    
            if image_counter is False:
                image_no = 1
                image_counter = True
                
            else:
                image_no = image_no + 1
    
            if liste_erstellt is False:
                liste = []
                entry = {
                    "image id" : image_id,
                    "image_no" : image_no,
                    "Numberplate_type" : type_of_np,
                    "np_identity" : img_buffer[1],
                    "image_file_name" : image_file_name,
                    "manual_name" : manual_name,
                    "effects" : effects
                    }
                liste.append(entry)
                liste_erstellt = True
                
            else:
                entry = {
                    "image id" : image_id,
                    "image_no" : image_no,
                    "Numberplate_type" : type_of_np,
                    "np_identity" : img_buffer[1],
                    "image_file_name" : image_file_name,
                    "manual_name" : manual_name,
                    "effects" : effects
                    }
                liste.append(entry)
                            
            #Path to the file in which the json file will be saved
            dateipfad = str(DIR["synthetic_pictures"] / f"holy_list.json")
    
            # Convert the list to JSON format and write it to a file
            # This step is repeated for every picture, to avoid loss of data in the event of a programme crash
            with open(dateipfad, 'w') as datei:
                json.dump(liste, datei)
            
#Path to the file in which the json file will be saved
dateipfad = str(DIR["synthetic_pictures"] / f"holy_list.json")

#Convert the list to JSON format and write it to a file
with open(dateipfad, 'w') as datei:
    json.dump(liste, datei)

