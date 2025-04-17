# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:59:50 2024
@author: LarsG

Three different functions to create more realistic synthetic plates:
    - apply_fog works good for specific values. But it can lead to a high level 
    of transparency of the synthetic numberplate. In this case the original numberplate comes into forefront. 
    - apply_dust do not simulate real dust -> should be improved
    - apply_darkness works good for specific values.

"""

import cv2
import numpy as np

#Function to apply fog to the synthetic numberplate
def apply_fog(raw_numberplate, numberplate_intensity = 1, fog_intensity=0.5, grey_value = 255):
    #Generation of a fog haze (Matrix with the same size as the synthetic numberplate)
    fog = np.ones_like(raw_numberplate) * grey_value  #Specify the grey value for the colour of the fog

    #Mix the fog haze with the synthetic numberplate
    result = cv2.addWeighted(raw_numberplate, numberplate_intensity, fog, fog_intensity, 0) # 0-value is Gamma-value
    return result

#Function to apply dust to the synthetic numberplate
#by calling this functions the existing pixel on the numberplates get replaced 
#randomly by pixels in the color 0-256
def apply_dust(raw_numberplate):

    noise = np.random.randint(0, 255, raw_numberplate.shape, dtype=np.uint8)
    result = cv2.add(raw_numberplate, noise)
    return result

#Function to apply darkness or brightness to the synthetic numberplate
def apply_darkness(raw_numberplate, brightness = 50):

    #result = np.where((255 - raw_numberplate) < brightness, 255, raw_numberplate + brightness)
    result = np.clip(raw_numberplate.astype(np.int16) + brightness, 0, 255).astype(np.uint8) 
    #result = np.clip(raw_numberplate.astype(np.uint8) + brightness, 0, 255).astype(np.uint8)
    return(result)
