# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:07:17 2023
@author: LarsG, updated by Tom-Marvin Rathmann

This function creates the synthetic numberplate.
By calling the function "generate_digits" from the file "Random_Digits.py", 
the letters and numbers for the synthetic numberplates are called.

As of March 2024 the following type of numberplates are supported:
    - single-line numberplate (fully implemented)
    - two-line numbmerplate for car/truck (fully implemented)
    - two-line numberplate for motorbikes (fully implemented, but with space for improvement)

Update (16.12.2024):
Author: Tom-Marvin Rathmann
- The function `get_numberplate_color` has been added. It identifies the most prominent bright color 
  in the number plate. Applying this color to the synthetic number plate background enhances its realism.
- File path handling has been integrated into the existing code.
"""

from PIL import Image, ImageDraw, ImageFont
import random
import string
import numpy as np

from Random_Digits import generate_digits
from Recognition_Tuv_Sticker import generate_tuv_sticker, generate_registration_sticker
from collections import Counter
from config import DIR


def get_numberplate_color(image, np_x_coordinate_list, np_y_coordinate_list):
    """
    Extracts the most prominent color from the bright pixels of a number plate in an image.

    This function crops the image based on the provided coordinates of the number plate, converts 
    the cropped image to grayscale for brightness calculation, and identifies based on the pixel ccordinates
    the most prominent color in the bright pixels by applying a brightness threshold.

    Parameters:
    image (PIL.Image.Image): The original image containing the number plate.
    np_x_coordinate_list (list): A list of x-coordinates defining the corners of the number plate.
    np_y_coordinate_list (list): A list of y-coordinates defining the corners of the number plate.

    Returns:
    tripel: The RGB values of the most common color found in the bright pixels of the number plate.
    """

    #Get the minimum and maximum coordinates from the given lists of x and y coordinates for the number plate
    x_start, x_end = min(np_x_coordinate_list), max(np_x_coordinate_list)
    y_start, y_end = min(np_y_coordinate_list), max(np_y_coordinate_list)

    #Crop the image based on the coordinates
    cropped_image = image.crop((x_start, y_start, x_end, y_end))

    #Convert the cropped image to grayscale for brightness identification
    cropped_image_gray = cropped_image.convert('L')
    cropped_image_array_gray = np.array(cropped_image_gray)

    #Find the maximum brightness value and set the threshold to 80% of that value
    max_luminance = np.max(cropped_image_array_gray)
    brightness_threshold = 0.8 * max_luminance

    #Create a mask to select only bright pixels based on the threshold
    bright_pixels_mask = cropped_image_array_gray >= brightness_threshold

    #Extract the bright pixels from the original cropped image
    cropped_image_rgb = cropped_image.convert('RGB')
    cropped_image_array_rgb = np.array(cropped_image_rgb)

    #Apply the mask to get only the RGB values of the bright pixels
    bright_pixels = cropped_image_array_rgb[bright_pixels_mask]

    #Count the occurrences of each color in the bright pixels
    color_counts = Counter(map(tuple, bright_pixels))

    #Get the most common color
    most_common_color = color_counts.most_common(1)[0][0]

    return most_common_color



def create_dummy_plate(width_NP = 521, height_NP = 111):
    # Creation of the image and the drawing area
    image = Image.new("RGBA", (width_NP, height_NP),(255,255,255,0))
    
    return(image)



def generate_number_plate(line_count=1, view = "rear", typ ="car", vehicle_image=None, np_x_coordinate_list=[], np_y_coordinate_list=[]):
    width_singleline_NP = 521
    height_singleline_NP = 111

    if typ == "motorbike":
        width_twoline_NP = 280
        height_twoline_NP = 200 
    else:
        width_twoline_NP = 340
        height_twoline_NP = 200
    
    font_size = 104
    country_size_singleline = 124
    country_size_twoline = 110
    

    #Load the different fonts to the file
    
    #This is the standard font for most of the numberplates 
    #(single-line numberplates and less than 8 digits (=maximum 7 digits))
    ##For two-line numberplates it depends on the number of digits in the specific area.
    text_font_middle = ImageFont.truetype(str(DIR["config"] / f"Europlate.TTF"), size=font_size)
    
    #This font is used for numberplates which have more than 7 digits.
    #For two-line numberplates it depends on the number of digits in the specific .
    text_font_close = ImageFont.truetype(str(DIR["config"] / f"Europlate_new.TTF"), size = font_size)
    
    #For the left part of the numberplate (blue background and letter D and EU circle) 
    #a different size is needed. Therefore the same font with different size is loaded.
    country_font_singleline = ImageFont.truetype(str(DIR["config"] / f"Europlate.TTF"), size=country_size_singleline)
    
    #The same as the previous comment + because two-line numberplate have a smaller size than single-line numberplate
    country_font_twoline = ImageFont.truetype(str(DIR["config"] / f"Europlate.TTF"), size=country_size_twoline)
   

    #####Call of function - Random_Digits - Creation of digits for the numberplate
    composition_variables,digits, number_digits, composition, len_first_block = generate_digits()
    #print("composition", composition_variables,"\n digits", digits, "\n number digits", number_digits, "\n composition", composition)
    
    #The EU circle and country letter "D" for the left part of the numberplate
    #is given in the bolt by calling "!"
    text_country = "!" 
    
    #The circles for the registration sticker and the TÜV sticker 
    #are given by the bolt and by calling ":"
    text_circle = ":"
    
    # Initialisation of the counters
    number_letters = 0
    number_numbers = 0

    # Run through the list and count the letters and numbers
    for element in digits:
        if isinstance(element, str) and element.isalpha():
            number_letters += 1
    number_numbers = len(digits)-number_letters
    # print(digits)
    # print("\n Anzahl Buchstaben", number_letters, "\n", "Anzahl Zahlen", number_numbers, "\n")

    width_sticker = 45
    begin_top = 12.5
    #begin_left = 58.5
    #letter_width = 47.5
    #number_width = 44.5
    distance_between1 = 9
    distance_between3 = 24

    #begin_left_circle = 168
    #begin_left_circle = 45 + distance_between1 + letter_width * composition["district"] + (distance_between1 * (composition["district"] -1) )
    begin_left_country = -6
    
    # Call functions to get registration sticker and TÜV sticker
    registration_sticker = generate_registration_sticker(str(DIR["config"] / f"Registration_Sticker_BW.png"))
    tuv_sticker = generate_tuv_sticker(str(DIR["config"] / f"HU-Plakette.png"))
    np_color = get_numberplate_color(vehicle_image, np_x_coordinate_list, np_y_coordinate_list)


    #This is the code to place the digits on the single-line numberplate
    if line_count == 1:
        image = create_dummy_plate(width_singleline_NP, height_singleline_NP)
        # # draw = create_dummy_plate()[1]
        
        # image = Image.new("RGBA", (521, 111), (255, 255, 255, 0))
        
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width_singleline_NP-1, height_singleline_NP-1), fill=np_color, outline="black", width=5, radius=10.2)
        #image.show()
        #image.save("C:/Users/LarsG/OneDrive/Master Aalen/Masterthesis/Python/CCDS/numberplate_raw.png") #RGBA-Format
        
        
        #Specify the distances and the fonts depending on the number of digits on the numberplate
        if number_digits > 7:
            text_font = text_font_close
            letter_width = 40.5
            number_width = 38.5
            distance_between2 = 63.5
        else:
            text_font = text_font_middle
            letter_width = 47.5
            number_width = 44.5
            distance_between2 = 67.5
        
        # This calculation is needed to have the same "free distance" at the start and at the end of the numberplate
        free_length_begin = (width_singleline_NP - 10 - 40 - distance_between2 - (number_letters*letter_width) - 
                             (number_numbers*number_width) - (distance_between1*(len(digits)-3)) - 
                             distance_between3)/2
        #print("\n free_length_begin", free_length_begin, "\n")  
    
      
        ### Now things are getting confusing :-)
        begin_left_circle = 40 + free_length_begin + letter_width * composition["district"] + (distance_between1 * (composition["district"]-1) )
        begin_left = begin_left_circle - letter_width * composition["district"]-(distance_between1 * (composition["district"]-1))
        begin_recognition = begin_left_circle + distance_between2

        #Insert Registration-Sticker
        image.paste(registration_sticker, (int(begin_left_circle + distance_between2/2-48/2+distance_between1/2), int(35 + begin_top+5)),registration_sticker)  #+9 bzw 9/2
    
        #Insert TÜV-Sticker
        if view == "rear":
            image.paste(tuv_sticker, (int(begin_left_circle + distance_between2/2-35/2+distance_between1/2), int(begin_top)), tuv_sticker)    #+9 bzw 9/2   
        else:
            None
        
        image.save(str(DIR["input"] / f"numberplate_buffer.png")) #RGBA-Format

        #Single-line numberplate with one district letter
        if composition["district"] == 1:
            #district digits
            draw.text((begin_left, begin_top), digits[0], fill="black", font=text_font)
            #recognition digits
            
            draw.text((begin_recognition, begin_top), digits[1], fill="black", font=text_font)
            draw.text(((begin_recognition+ letter_width +distance_between1), begin_top), digits[2], fill="black", font=text_font)
            distance = begin_recognition+ letter_width +distance_between1
            if digits[2].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
            width = number_width
            if number_digits >3:           
                draw.text(((distance + width + distance_between), begin_top), digits[3], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[3].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1    
                width = number_width  
                if number_digits >4:
                    draw.text(((distance + (width+ distance_between1)), begin_top), digits[4], fill="black", font=text_font)
                    distance = distance + (width+ distance_between1)
                    if digits[4].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                    width = number_width
                    if number_digits >5:
                        draw.text(((distance + (width+ distance_between1)), begin_top), digits[5], fill="black", font=text_font)
                        distance = distance + (width+ distance_between1)
                        if number_digits >6:
                            draw.text(((distance+number_width+distance_between1), begin_top), digits[6], fill="black", font=text_font)
                            distance = distance + number_width+distance_between1
                            if number_digits >7:
                                draw.text(((distance+(number_width+distance_between1)), begin_top), digits[7], fill="black", font=text_font)
            
        #Single-line numberplate with two district letters
        elif composition["district"] == 2:
            #district digits
            draw.text((begin_left, begin_top), digits[0], fill="black", font=text_font)
            draw.text(((begin_left + letter_width + distance_between1), begin_top), digits[1], fill="black", font=text_font)
            #recognition digits
            draw.text((begin_recognition, begin_top), digits[2], fill="black", font=text_font)
            draw.text(((begin_recognition + letter_width + distance_between1), begin_top), digits[3], fill="black", font=text_font)
            distance = begin_recognition + letter_width + distance_between1
            if digits[3].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
            width = number_width
            if number_digits >4:           
                draw.text(((distance + width + distance_between), begin_top), digits[4], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[4].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1
                width = number_width  
                if number_digits >5:
                    draw.text(((distance + (width + distance_between)), begin_top), digits[5], fill="black", font=text_font)
                    distance = distance + (width + distance_between) 
                    if digits[5].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                    width = number_width
                    if number_digits >6:
                        draw.text(((distance + distance_between + width), begin_top), digits[6], fill="black", font=text_font)
                        distance = distance + distance_between + width
                        if number_digits >7:
                            draw.text(((distance + number_width+distance_between1), begin_top), digits[7], fill="black", font=text_font)

            
        #Single-line numberplate with three district letters
        else:
            #district digits
            draw.text((begin_left, begin_top), digits[0], fill="black", font=text_font)
            draw.text(((begin_left + letter_width + distance_between1), begin_top), digits[1], fill="black", font=text_font)
            draw.text(((begin_left + letter_width + distance_between1 + letter_width + distance_between1), begin_top), digits[2], fill="black", font=text_font)
            #recognition digits
            draw.text((begin_recognition, begin_top), digits[3], fill="black", font=text_font)
            draw.text(((begin_recognition + letter_width + distance_between1), begin_top), digits[4], fill="black", font=text_font)
            distance = begin_recognition + letter_width + distance_between1
            if digits[4].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
            width = number_width
            if number_digits >5:           
                draw.text(((distance + width + distance_between), begin_top), digits[5], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[5].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1
                width = number_width  
                if number_digits >6:
                    draw.text(((distance+ (width + distance_between1)), begin_top), digits[6], fill="black", font=text_font)
                    distance = distance + width + distance_between1
                    if digits[6].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                    width = number_width
                    if number_digits >7:
                        draw.text(((distance + (width + distance_between1)), begin_top), digits[7], fill="black", font=text_font)
                        
        #draw.text((begin_left_circle+9, begin_top), text_circle, fill="black", font=text_font)
        draw.text((begin_left_country, 5), text_country, fill="blue", font=country_font_singleline)

##########################################################################################################################################

    #The following section is to create a two-line-numberplate
    elif line_count == 2:
        
        #Crfeation of a raw two-line numberplate to write the digits on it
        image = create_dummy_plate(width_twoline_NP, height_twoline_NP)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width_twoline_NP-1, height_twoline_NP-1), fill=np_color, outline="black", width=5, radius=10.2)
      
        #Define where the start of the digits is.
        begin_top = 12.5
        begin_top_line2 = height_twoline_NP/2+7.5
        distance_between1 = 9
        distance_between3 = 24

        begin_left_country = -6
        
        #Calculation of the distance at the start and the end of first line of the numberplate
        free_length_top = width_twoline_NP - 10 -40
        
        #Calculation of the distance at the start and the end of second line of the numberplate
        free_length_bottom = width_twoline_NP - 10
        
        #Definition of the fontsize based on the number of digits in the first block (registration letters)
        #and the width of the raw numberplate
        if number_digits - len_first_block > 5 or width_twoline_NP == 280:
            text_font = text_font_close
            letter_width = 40.5
            number_width = 38.5
        #If nothing special for the numberplate, the font isn't special too ;-)
        else:
            text_font = text_font_middle
            letter_width = 47.5
            number_width = 44.5
        
        #Also some calculations of distances is needed due to the different requirements 
        #coming from the standard and depending on the composition of the digits of the numberplate
        if len_first_block == 1:
            distance_between2 = 25
            distance_between0_t = (free_length_top-width_sticker-distance_between2-letter_width)/2
        elif len_first_block == 2:
            distance_between2 = 16
            distance_between0_t = (free_length_top-width_sticker-distance_between2-distance_between1-letter_width*2)/2
        else:
            distance_between2 = 8
            distance_between0_t = (free_length_top-width_sticker-distance_between2-distance_between1*2-letter_width*3)/2
        
        xi = composition['recognition_letter']        
        yi = composition['number']
        
        distance_between0_b = (free_length_bottom - xi*letter_width - distance_between3 - yi*number_width-(xi-1)*distance_between1-(yi-1)*distance_between1)/2
        
        #Insert Registration-Sticker
        image.paste(registration_sticker, (int(width_twoline_NP-5-distance_between0_t-width_sticker/2), int(35 + begin_top+5)),registration_sticker)  #+9 bzw 9/2
    
        #Insert TÜV-Sticker (differenzation for front and rear numberplate)
        if view == "rear":
            image.paste(tuv_sticker, (int(width_twoline_NP-5-distance_between0_t-35/2), int(begin_top)), tuv_sticker)    #+9 bzw 9/2   
        else:
            None
                
        #begin_left_circle = 40 + distance_between1 + letter_width * composition["district"] + (distance_between1 * (composition["district"]-1) )
        begin_left_top = 5 + 40 + distance_between0_t
        begin_left_bottom = 5 + distance_between0_b
        

        #two-line numberplate with one district letter
        if composition["district"] == 1:
            #district digits
            draw.text((begin_left_top, begin_top), digits[0], fill="black", font=text_font)
            #recognition digits
            draw.text((begin_left_bottom, begin_top_line2), digits[1], fill="black", font=text_font)
            draw.text(((begin_left_bottom + letter_width +distance_between1), begin_top_line2), digits[2], fill="black", font=text_font)
            distance = begin_left_bottom+ letter_width +distance_between1
            if digits[2].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
            width = number_width
            if number_digits >3:           
                draw.text(((distance + width + distance_between), begin_top_line2), digits[3], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[3].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1    
                width = number_width  
                if number_digits >4:
                    draw.text(((distance + (width+ distance_between1)), begin_top_line2), digits[4], fill="black", font=text_font)
                    distance = distance + (width+ distance_between1)
                    if digits[4].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                    width = number_width
                    if number_digits >5:
                        draw.text(((distance + (width+ distance_between1)), begin_top_line2), digits[5], fill="black", font=text_font)
                        distance = distance + (width+ distance_between1)
                        if number_digits >6:
                            draw.text(((distance+number_width+distance_between1), begin_top_line2), digits[6], fill="black", font=text_font)
                            distance = distance + number_width+distance_between1
                            if number_digits >7:
                                draw.text(((distance+(number_width+distance_between1)), begin_top_line2), digits[7], fill="black", font=text_font)
            
        #two-line numberplate with two district letters
        elif composition["district"] == 2:
            #district digits
            draw.text((begin_left_top, begin_top), digits[0], fill="black", font=text_font)
            draw.text(((begin_left_top + letter_width + distance_between1), begin_top), digits[1], fill="black", font=text_font)
            #recognition digits
            draw.text((begin_left_bottom, begin_top_line2), digits[2], fill="black", font=text_font)
            draw.text(((begin_left_bottom+letter_width+distance_between1),begin_top_line2),digits[3], fill = "black", font = text_font)
            distance = begin_left_bottom + letter_width + distance_between1
            if digits[3].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
                width = number_width
            if number_digits >4:           
                draw.text(((distance + width + distance_between), begin_top_line2), digits[4], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[4].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1
                width = number_width  
                if number_digits >5:
                    draw.text(((distance + (width + distance_between)), begin_top_line2), digits[5], fill="black", font=text_font)
                    distance = distance + (width + distance_between) 
                    if digits[5].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                        width = number_width
                    if number_digits >6:
                        draw.text(((distance + distance_between + width), begin_top_line2), digits[6], fill="black", font=text_font)
                        distance = distance + distance_between + width
                        if number_digits >7:
                            draw.text(((distance + number_width+distance_between1), begin_top_line2), digits[7], fill="black", font=text_font)

            
        #two-line numberplate with three district letters
        else:
            #district digits
            draw.text((begin_left_top, begin_top), digits[0], fill="black", font=text_font)
            draw.text(((begin_left_top + letter_width + distance_between1), begin_top), digits[1], fill="black", font=text_font)
            draw.text(((begin_left_top + letter_width + distance_between1 + letter_width + distance_between1), begin_top), digits[2], fill="black", font=text_font)
            #recognition digits
            draw.text((begin_left_bottom, begin_top_line2), digits[3], fill="black", font=text_font)
            draw.text(((begin_left_bottom + letter_width + distance_between1), begin_top_line2), digits[4], fill="black", font=text_font)
            distance = begin_left_bottom + letter_width + distance_between1
            if digits[4].isalpha() == True:
                width = letter_width
                distance_between = distance_between3
            else:
                distance_between = distance_between1
                width = number_width
            if number_digits >5:           
                draw.text(((distance + width + distance_between), begin_top_line2), digits[5], fill="black", font=text_font)
                distance = distance + width + distance_between
                if digits[5].isalpha() == True:
                    width = letter_width
                    distance_between = distance_between3
                else:
                    distance_between = distance_between1
                    width = number_width  
                if number_digits >6:
                    draw.text(((distance+ (width + distance_between1)), begin_top_line2), digits[6], fill="black", font=text_font)
                    distance = distance + width + distance_between1
                    if digits[6].isalpha() == True:
                        width = letter_width
                        distance_between = distance_between3
                    else:
                        distance_between = distance_between1
                        width = number_width
                    if number_digits >7:
                        draw.text(((distance + (width + distance_between1)), begin_top_line2), digits[7], fill="black", font=text_font)
                        
        #draw.text((begin_left_circle+9, begin_top), text_circle, fill="black", font=text_font)
        draw.text((begin_left_country, 5), text_country, fill="blue", font=country_font_twoline)
        
    #     pass
    # else:
    #     print("Ungültige Anzahl von Linien")
    
    #image.show()  # Zeige das generierte Bild an
    
    return(image,digits)


#generate_number_plate(1,typ = "no motorbike")
