# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:48:35 2024
@author: LarsG

This function creates the registration and TÜV sticker.
As of March 2024 only the registration sticker for Baden-Württemberg, countrystate Stuttgart is implemented.
For the TÜV sticker always the same orientation (12 at the top) is used. 
"""
import random
from PIL import Image


def generate_registration_sticker(path):
    #Import of the registration sticker
    registration_sticker = Image.open(path).convert("RGBA")
    #Define the size of the sticker to the nominal size which is given by the standard
    registration_sticker = registration_sticker.resize((48,48), Image.BILINEAR)
    #registration_sticker.show()
    return(registration_sticker)


def generate_tuv_sticker(path):
    tuv_sticker = Image.open(path)
    
    #These colors are given for the TÜV stickers (black is not included)
    colors = [
        (204, 185, 141, 255),  # RAL 1012 - yellow
        (0, 92, 169, 255),     # RAL 5015 - blue
        (217, 96, 59, 255),    # RAL 2000 - organge
        (79, 122, 70, 255),    # RAL 6018 - green
        (201, 81, 108, 255),   # RAL 3015 - pink
        (127, 66, 37, 255),    # RAL 8004 - brown
        (0, 0, 0, 255)         # RAL 9005 - black
    ]
    
    counter = random.randint(0,5)
    #Size of the image
    weidth, height = tuv_sticker.size
    
    #Go through the pixels and replace the white areas with the replacement colour
    for x in range(weidth):
        for y in range(height):
            pixel = tuv_sticker.getpixel((x, y))  #Retrieving pixel values (R,G,B)
            
            if all(abs(p - 255) < 10 for p in pixel):
                tuv_sticker.putpixel((x, y), colors[counter % len(colors)]) 
                #tuv_sticker.putpixel((x, y), colors[1]) 
    
    #Define the size of the sticker to the nominal size which is given by the standard
    tuv_sticker = tuv_sticker.resize((35,35),Image.BILINEAR)
    
    return(tuv_sticker)