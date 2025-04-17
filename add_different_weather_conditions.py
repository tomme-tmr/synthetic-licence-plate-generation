'''
Created on Jul 15 2024
@author: Tom-Marvin Rathmann updated by Tom-Marvin Rathmann

This module provides a set of functions for enhancing vehicle images
by applying various visual effects, simulating challenging environmental conditions.
It is designed to be integrated into an image processing pipeline, enabeling the addition
of artifacts such as darkness, weather effects and specific plate effects to expand the number 
of an existing dataset of images.

Functions:
- set_effect_styles: Randomly selects and configures light SETTING, plate, and weather effects to apply to an image.
- generate_depth_map: Generating a depth map of the image to enable depth based weather effect creation.
- apply_darkness: Reduces brightness of the image and optionally adds headlight effects based on certain conditions.
- add_headlight_effect: Generates headlight glows and rays to simulate vehicle headlights in low visibility.
- add_weather_effect: Applies various weather effects like snow, rain, fog, and sun reflections to the image.
- add_plate_specific_effect: Adds specific effects (dirt, snow, shadow) to simulate real-world conditions on vehicle license plates.

Usage:
This module is intended to be used in image processing applications,
where simulating environmental effects on vehicle images is required,
such as on machine learning datasets for licens plate extraction, autonomous vehicles
or other related image augmentation tasks.

Dependencies:
- Pillow (for image processing)
- NumPy (for numerical operations)
- Perlin noise implementation (for realistic noise generation)
- OpenCV (for reading images)
- torch (For loading and running the pre-trained MiDaS depth estimation model)

Update (16.12.2024):
Author: Tom-Marvin Rathmann
- The calculation and usage of a depth map for the image was added to enhance the effect realism.
- The "set_effect_styles" function was updated to match the new effects dictionary
'''


import noise
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps




def set_effect_styles():
    '''
    Sets random effect styles for an image processing pipeline, including light settings, plate effects, 
    and weather effects. The function is usig different propabilities to simulate different conditions.

    This function generates a dictionary of effect styles, which includes settings for:
    - Light: Defines the light condition ('bright' or 'dark') applied to the image.
    - Plate: Effects applied to a license plate (e.g., 'dirt', 'snow', 'shadow').
    - Weather: Weather-related effects (e.g., 'snow', 'rain', 'fog', 'sun_reflection').

    The function uses probabilities to simulate random changes in these conditions.

    Returns:
    dict: A dictionary containing the following keys:
        - "light": A dictionary with two keys:
            - "effect": A string defining the light setting, either 'bright' or 'dark', selected randomly with a 70% and 30% probability.
            - "strength": A placeholder for light intensity. Is needet in the pipeline but will not be defined in this fuction.
        - "plate": A dictionary with two keys:
            - "effect": A list of effects that apply to the license plate, which can include 'dirt', 'snow', and 'shadow'.
            - "strength": A placeholder for the effect intensity on the plate. Is needet in the pipeline but will not be defined in this fuction.
        - "weather": A dictionary with two keys:
            - "effect": A list of weather effects (e.g., 'snow', 'rain', 'fog', 'sun_reflection') based on the light setting.
            - "strength": A placeholder for the effect intensity on the plate. Is needet in the pipeline but will not be defined in this fuction.

    The function uses specific probabilities for each effect:
    - Light effect ('bright' or 'dark') has a 70% and 30% chance.
    - Plate effects (e.g., 'dirt', 'snow', 'shadow') are applied with specific probabilities (20% for dirt/snow (dirt is more likely then snow), 20% for shadow if 'bright').
    - Weather effects are selected with a 30% probability, based on the light setting. Different combinations of weather effects are possible for both 'bright' and 'dark' light settings
    '''

    #Initialize the effects dictionary with default values
    effects = {
    "light": {
        "effect": None,
        "strength": None
    },
    "plate": {
        "effect": None,
        "strength": None
    },
    "weather": {
        "effect": None,
        "strength": None
    },
    "headlights": {
        "mode": 'random'
    }
    }


    ###Set light setting effect
    #Randomly choose between 'bright' and 'dark' with different probabilities of 70% and 30%
    effects["light"]["effect"] = np.random.choice(['bright','dark'], p=[0.7, 0.3])

    ###Set plate effect
    plate_effect = []

    #With a probability of 20%, add a 'dirt' or 'snow' effect to the plate effect list
    if np.random.rand() < 0.2:

        #Randomly choose between 'dirt' and 'snow' with different probabilities of 70% and 30%
        plate_effect.append(np.random.choice(['dirt','snow'], p=[0.7, 0.3]))
    else:
        pass
    
    #If the light setting is 'bright', there is a 20% chance to add a 'shadow' effect to the plate
    if effects["light"]["effect"] == 'bright' and np.random.rand() < 0.2:
        plate_effect.append('shadow')
    else:
        pass

    effects["plate"]["effect"] = plate_effect

    
    ###Set weather effect
    #Possible combinations of weather effects for a 'bright' light setting
    bright_effect_combinations = [
        ['snow'],
        ['rain'],
        ['fog'],
        ['sun_reflection'],
        ['snow', 'fog'],
        ['snow', 'sun_reflection'],
        ['rain', 'fog'],
        ['sun_reflection','fog'],
        ['sun_reflection','snow', 'fog']
    ]

    #Possible combinations of weather effects for a 'dark' light setting
    dark_effect_combinations = [
        ['snow'],
        ['rain'],
        ['fog'],
        ['snow', 'fog'],
        ['rain', 'fog']
    ]

    #With a probability of 30%, select a random combination of weather effects based on the light setting condition
    if effects["light"]["effect"] == 'bright' and np.random.rand() < 0.3:
        index_temp = np.random.choice(list(range(len(bright_effect_combinations))))
        weather_effect = bright_effect_combinations[index_temp]
    elif effects["light"]["effect"] == 'dark' and np.random.rand() < 0.3:
        index_temp = np.random.choice(list(range(len(dark_effect_combinations))))
        weather_effect = dark_effect_combinations[index_temp]
    else:
        weather_effect = []

    effects["weather"]["effect"] = weather_effect

    return effects
    


def generate_depth_map(image_path):
    '''
    Generates a depth map from an input image using a pre-trained MiDaS model.

    This function loads a pre-trained MiDaS model which is be used to estimate the
    depth of an provided image. The output is a depth map array where 
    each pixel value represents the relative distance of the corresponding point in the image 
    from the camera. The depth map is useful for applying depth based effects such as fog.

    Parameters:
    image_path (str): The file path to the image where the depth map should be calculated for.

    Returns:
    numpy.array: Array containing the estimated depth for each pixel of the input image.

    Implementation is based on the Pytorch documentation: https://pytorch.org/hub/intelisl_midas_v2/
    '''

    #Load pre-trained MiDaS model
    #model_type = "DPT_Large"
    #model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"  

    #Load MiDaS model from torch hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    #Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    #Set model to evaluation mode
    midas.eval()
    
    #Load necessary transformations for the model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    #Choose transformation based on the model type
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    #Load and preprocess the input image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Apply the transformation to prepare the image for the model
    input_batch = transform(img).to(device)

    #Perform inference to generate the depth map
    with torch.no_grad():
        prediction = midas(input_batch)

        #Upsample the predicted depth map to match the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    #Convert the depth map prediction to a numpy array for visualization
    output = prediction.cpu().numpy()

    return output




def add_headlight_effekt(draw, x, y, np_width, np_height, image_width, image_height):
    """
    Adds a headlight effect to an image layer.

    This function draws a headlight effect at specified coordinates on the given image layer. 
    It simulates a headlight by creating bright elliptical spots with gradient transparency and 
    adding radial light rays from the headlight center.

    Parameters:
    draw (ImageDraw.Draw): The image layer.
    x (int): The x-coordinate of the number plate center.
    y (int): The y-coordinate of the number plate center.
    np_width (int): The width of the number plate.
    np_height (int): The height of the number plate.
    image_width (int): The width of the image layer.
    image_height (int): The height of the image layer.

    Returns:
    ImageDraw.Draw: The image layer with the added headlight effect.
    """
    
    #Calculation of two headlight centers positions
    x1 = x + 1.25 * np_width 
    y1 = y - 2 * np_height
    x2 = x - 1.25 * np_width 
    y2 = y - 2 * np_height

    #List of headlight centers
    points = [(x1, y1),(x2, y2)]

    #Iteration through the List of headlight centers
    for point in points:
        x,y = point
        headlight_radius = 3 * np_height

        #Draw the headlight circles
        for radius in reversed(range(headlight_radius)):
            
            #Set the alpha transparency based on the radius
            if radius < 2/3 * headlight_radius:
                alpha = 255
            else:
                alpha = int(255 * (1 - (radius - 2/3*headlight_radius) / (1/3*headlight_radius)))
            color = (255, 255, 255, alpha)

            #Draw the headlight circle with gradient transparency
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        #Add light rays from the headlight center
        sun_center_x, sun_center_y = x, y
        num_rays = 15
        max_ray_length = int(np.sqrt(image_width**2 + image_height**2) / 2)
        
        #Randomly select the angle and length of each light ray
        for _ in range(num_rays):
            angle = np.random.uniform(0, 2 * np.pi)
            ray_length = np.random.uniform(3*headlight_radius, max_ray_length)
            end_x = int(sun_center_x + ray_length * np.cos(angle))
            end_y = int(sun_center_y + ray_length * np.sin(angle))

            #Add the light rays the headlight effect image layer
            draw.line((sun_center_x, sun_center_y, end_x, end_y), fill=(255, 255, 255, 255), width=7)
    
    return draw


def apply_darkness(image, depth_map, np_center_x, np_center_y, np_width, np_height, darkness_strength, headlights_mode):
    '''
    Applies a darkness effect to a given image to simulate low-light conditions.

    This function reduces the brightness of the input image to a random level between 8% and 50% 
    of the original brightness. If the brightness level is very low (below 25%), it additionaly
    calls the 'add_headlights_effect' funkction by a 70% chance.

    Parameters:
    image (PIL.Image): The input image to be processed.
    np_center_x (int): The x-coordinate of the center of the number plate.
    np_center_y (int): The y-coordinate of the center of the number plate.
    np_width (int): The width of the number plate.
    np_height (int): The height of the number plate.
    darkness_strength (str): Variable for the selection of the effect strength. Options: 'subtle','mild','moderate','strong','intense', None
    headlights_mode (str): Variable to select if a headlights effect should be added. Options: 'on', 'off', 'random'

    Returns:
    PIL.Image: The processed image with darkness and optional headlight effect.
    '''

    #Converts the image to RGBA mode to allow transparency handling
    image = image.convert('RGBA')
    width, height = image.size

    #Normalize the depth map to range from 0 (near) to 1 (far)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    #Set the brightness level based on the input variable
    #TMR####brightness_level_selection = {"subtle": 0.5, "mild": 0.395, "moderate": 0.29, "strong": 0.185, "intense": 0.08}
    brightness_level_selection = {"subtle": 0.5, "mild": 0.4125, "moderate": 0.325, "strong": 0.2375, "intense": 0.15}
    if darkness_strength in brightness_level_selection:
        brightness_level = brightness_level_selection[darkness_strength]
    else:
        #Randomly selects a brightness level between 0.15 and 0.5
        brightness_level = np.random.uniform(0.08, 0.5)
    
    #Apply the brightness reduction
    enhancer = ImageEnhance.Brightness(image)
    output = enhancer.enhance(brightness_level)
    
    #Apply darkness adjustment based on the depth map
    output_pixels = output.load()  #Load the pixel data of the image
    for i in range(width):
        for j in range(height):
            #Depth influences the darkness level (farther = darker)
            depth_value = depth_map_normalized[j, i]
            depth_influence = (1 - depth_value) * 0.1
            
            # Modify the brightness based on depth influence
            pixel = output_pixels[i, j]
            r, g, b, a = pixel
            r = max(0, r - int(depth_influence * 255))
            g = max(0, g - int(depth_influence * 255))
            b = max(0, b - int(depth_influence * 255))
            #Update the pixel
            output_pixels[i, j] = (r, g, b, a)


    #Add headlight effect if selected or add it randomly by a chance of 70% if the brightness level is less than 0.25 (25%)
    if headlights_mode == 'on' or (headlights_mode == 'random' and brightness_level < 0.25 and np.random.rand() < 0.7):
        #Creates a new layer for the headlight effect
        headlight_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        #Add the headlight effect to the layer
        draw = ImageDraw.Draw(headlight_layer)
        draw = add_headlight_effekt(draw, np_center_x, np_center_y, np_width, np_height, width, height)

        #Apply Gaussian blur to the headlight effect layer
        headlight_layer = headlight_layer.filter(ImageFilter.GaussianBlur(15))

        #Composite the headlight layer onto the darkened image
        output = Image.alpha_composite(output, headlight_layer)
    
    return output



def add_weather_effect(image, depth_map, effect_setting, effect_dict):
    '''
    Applies various weather effects to an image.

    This function adds specified weather effects such as snow, rain, fog, or sun reflection
    to a given image based on a list of effect types. Each effect type is applied based on the specified probabilities
    and characteristics.

    Parameters:
    image (PIL.Image): The input image to which the weather effects will be applied.
    effect_types (list of str): A list of weather effect types to apply (e.g., 'snow', 'rain', 'fog', 'sun_reflection'). ###TMR -> change
    effect_setting (str): Describes if the effects should be set random or are pre set by the User.

    Returns:
    PIL.Image: The processed image with the specified weather effects applied.
    '''

    #Convert the image to RGBA mode to handle transparency
    image = image.convert('RGBA')
    width, height = image.size

    effect_types = effect_dict['weather']['effect']
    effect_strength = effect_dict['weather']['strength']

    #Iterating through the list of effect types to applie multiple effects if nessecary
    for effect_type in effect_types:

        if effect_type == 'snow':

            #Sets the snow density based on the selected strength in the config file or a random number
            snow_density_selection = {"subtle": 0.02, "mild": 0.0325, "moderate": 0.045, "strong": 0.0575, "intense": 0.07}
            if effect_setting == 'specified' and effect_strength in snow_density_selection:
                snow_density = snow_density_selection[effect_strength]
            else:
                #Randomly selects a density level between 0.02 and 0.7
                snow_density = np.random.uniform(0.02, 0.07)
            
            #Normalize the depth map to range from 0 (near) to 255 (far)
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX) 

            #Creates a layer for the snow effect
            snow_layer = Image.new('RGBA', (width, height), (255, 255, 255, 0))

            snow_layer_data = snow_layer.load()

            #Assign normalized depth values to the alpha channel (range 0-255)
            for y in range(height):
                for x in range(width):
                    #Get the normalized depth value
                    alpha_value = int(depth_map_normalized[y, x])

                    #Adding an effect strength to the depth_map value, which will be used as alpha channel (value will be between 0.65 and 0.95)
                    alpha_value = min(255, int((255 - alpha_value) * (1 - ((0.07 - snow_density) * 6.3)) * 0.999))

                    #Set depth_map value as alpha channel
                    snow_layer_data[x, y] = (255, 255, 255, alpha_value)

            snow_flake_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(snow_flake_layer)

            #Defines the number of snowflakes which should be added to the image
            num_snowflakes = int(width * height * snow_density*3)

            #Drawing an rectangle with different sizes for the number of snowflakes
            for _ in range(num_snowflakes):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                size = np.random.randint(1, 2)#4

                depth_snow_flake_value = depth_map_normalized[y, x]
                snow_flake_alpha = min(255, int(depth_snow_flake_value * 1.4))

                draw.rectangle([x, y, x + size, y + size], fill=(255, 255, 255, snow_flake_alpha))

            #Apply a slight blur to the snow layer for realism
            snow_flake_layer = snow_flake_layer.filter(ImageFilter.GaussianBlur(1))

            #Composite the snow flake layer onto the snow layer
            snow_layer = Image.alpha_composite(snow_layer, snow_flake_layer)

            #Composite the snow layer onto the image
            image = Image.alpha_composite(image, snow_layer)
        


        elif effect_type == 'rain':

            #Sets the rain density based on the selected strength in the config file or a random number
            rain_density_selection = {"subtle": 0.02, "mild": 0.0325, "moderate": 0.045, "strong": 0.0575, "intense": 0.07}
            if effect_setting == 'specified' and effect_strength in rain_density_selection:
                rain_density = rain_density_selection[effect_strength]
            else:
                #Randomly selects a density level between 0.02 and 0.7
                snow_density = np.random.uniform(0.02, 0.07)
            
            #Normalize the depth map to range from 0 (near) to 255 (far)
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX) 

            #Creates a layer for the snow effect
            rain_layer = Image.new('RGBA', (width, height), (40, 60, 72, 0))

            rain_layer_data = rain_layer.load()

            #Assign normalized depth values to the alpha channel (range 0-255)
            for y in range(height):
                for x in range(width):
                    #Get the normalized depth value
                    alpha_value = int(depth_map_normalized[y, x])

                    #Adding an effect strength to the depth_map value, which will be used as alpha channel
                    alpha_value = min(255, int((255 - alpha_value) * (1 - ((0.07 - rain_density) * 6.3)) * 0.95))

                    #Set depth_map value as alpha channel
                    rain_layer_data[x, y] = (40, 60, 72, alpha_value)

            #Creates a layer for the rain effect
            rain_drop_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(rain_drop_layer)

            #Defines the number of rain drops which should be added to the image
            num_raindrops = int(width * height * rain_density)

            #Drawing an vertical line with different length for each rain drop
            for _ in range(num_raindrops):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                length = np.random.randint(5, 20)

                depth_rain_drop_value = depth_map_normalized[y, x]
                rain_drop_alpha = min(255, int(depth_rain_drop_value * 1.4))

                draw.line([x, y, x, y + length], fill=(49, 60, 72, rain_drop_alpha), width=1)

            #Composite the rain drop layer onto the rain layer
            rain_layer = Image.alpha_composite(rain_layer, rain_drop_layer)

            #Composite the rain layer onto the image
            image = Image.alpha_composite(image, rain_layer)
        


        elif effect_type == 'fog':
            #Set effect strength option based on config file input or select randomly
            effect_strength_selection = {"subtle": 0, "mild": 17.5, "moderate": 35, "strong": 52.5, "intense": 70} 
            if effect_setting == 'specified':
                effect_strength_value = effect_strength_selection[effect_strength]
            else:
                #Define effect strength options and select one randomly with different possibilities (subtle = 30%, mild = 25%, moderate = 20%, strong = 15%, intense = 10%)
                effect_strength_value = np.random.choice(list(effect_strength_selection.values()), p=[0.3,0.25,0.2,0.15,0.1])

            #Normalize the depth map to range from 0 (near) to 255 (far)
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX) 

            #Creates a basic layer for the fog effect
            basic_fog_layer = Image.new('RGBA', (width, height), (255, 255, 255, 0))

            basic_fog_layer_data = basic_fog_layer.load()

            #Assign normalized depth values to the alpha channel (range 0-255)
            for y in range(height):
                for x in range(width):
                    #Get the normalized depth value
                    alpha_value = int(depth_map_normalized[y, x])

                    #Adding adjusting the depth_map effect
                    alpha_value = min(255, int(alpha_value * 0.8))

                    #Set depth_map value as alpha channel
                    basic_fog_layer_data[x, y] = (255, 255, 255, alpha_value)


            '''
            #Creates a basic layer for the fog effect to enhance the effect
            basic_fog_layer = Image.new('L', (width, height), 0)

            #Selecting randome pixel values using a normal distribution with mean 128 and standard deviation 30
            fog = np.random.normal(loc=128, scale=30, size=(height, width)).astype(np.uint8)
            basic_fog_layer = Image.fromarray(fog, mode='L')
            
            #Adds a blur to the base fog layer
            basic_fog_layer = basic_fog_layer.filter(ImageFilter.GaussianBlur(5))
            '''

            #Creates a layer for the additional fog effect (perlin noise)
            noise_image = np.zeros((height, width), dtype=np.float32)
            
            #Selecting a randome value in the ranges of the parameters for the perlin noise function
            scale = 800
            octaves = np.random.randint(3, 8)
            persistence = np.random.uniform(0.5, 0.7)
            lacunarity = np.random.uniform(2.0, 3.5)

            #Generate a perlin noise layer for each pixel in the image
            for y in range(height):
                for x in range(width):
                    nx = x / scale
                    ny = y / scale

                    #Calculate the noise values using the perlin noise function and the randomly selected parameters
                    noise_value = noise.pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=width, repeaty=height)
                    noise_image[y, x] = noise_value

            #Normalize the noise values to the range [0, 255]
            min_val = noise_image.min()
            max_val = noise_image.max()
            noise_image = (((noise_image - min_val) / (max_val - min_val) * 255)).astype(np.uint8)

            #Applying the additional strength effect to the noise values by reducing the pixel values
            fog = Image.fromarray(noise_image, mode='L')
            fog = fog.point(lambda p: p - effect_strength_value)

            #Create a second layer 
            fog_rgba = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            
            #Apply the perlin noise fog values as the alpha channel to the new RGBA image
            fog_rgba.putalpha(fog)
            
            #Apply a slight blur to the fog layer for realism
            fog_rgba = fog_rgba.filter(ImageFilter.GaussianBlur(5))
            
            #Composite the basic and perlin noise-based fog layers onto the image
            image = Image.composite(image, Image.new('RGBA', image.size, (255, 255, 255, 255)), fog_rgba)
            #test####mask = Image.new("L", image.size, 30)
            #test####image = Image.composite(image, fog_rgba, mask)
            image = Image.composite(image, Image.new('RGBA', image.size, (255, 255, 255, 255)), basic_fog_layer)


        
        elif effect_type == 'sun_reflection':
            num_reflections = 1
            
            #Creates a layer for sun reflection effect
            reflection_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(reflection_layer)

            
            for _ in range(num_reflections):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height/2)

                #Sets the sun radius based on the selected strength in the config file or a random number
                if effect_setting == 'specified':
                    if effect_strength == 'subtle':
                        max_radius = 150
                    elif effect_strength == 'mild':
                        max_radius = 175
                    elif effect_strength == 'moderate':
                        max_radius = 200
                    elif effect_strength == 'strong':
                        max_radius = 225
                    elif effect_strength == 'intense':
                        max_radius = 250
                else:
                    max_radius = np.random.randint(150, 250)
                
                #Draw the circles to simulate the sun
                for radius in reversed(range(max_radius)):

                    #Set the alpha transparency based on the radius
                    if radius < 1/2 * max_radius:
                        alpha = 255
                    else:
                        alpha = int(255 * (1 - (radius - 1/2*max_radius) / (1/2*max_radius)))
                    color = (255, 255, 255, alpha)

                    #Draw the sun circle with gradient transparency
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
                
                #Add light rays from the center of the sun
                sun_center_x, sun_center_y = x, y
                num_rays = 15
                max_ray_length = int(np.sqrt(width**2 + height**2) / 2)
                
                #Randomly select the angle and length of each light ray
                for _ in range(num_rays):
                    angle = np.random.uniform(0, 2 * np.pi)
                    ray_length = np.random.uniform(3*max_radius, max_ray_length)
                    end_x = int(sun_center_x + ray_length * np.cos(angle))
                    end_y = int(sun_center_y + ray_length * np.sin(angle))
                    draw.line((sun_center_x, sun_center_y, end_x, end_y), fill=(255, 255, 255, 255), width=10)
            
            #Apply a blur to the reflection layer for realism
            reflection_layer = reflection_layer.filter(ImageFilter.GaussianBlur(15))
            image = Image.alpha_composite(image, reflection_layer)
        
        else:
            image = image

    output = image
    return output





def add_plate_specific_effect(image, effect_setting, effect_dict):
    """
    Applies specific surface effects to a handed image (license plate).

    Parameters:
    image (PIL.Image): The input image on which to apply the surface effects (license plate image).
    plate_effects (list): List of effects to apply (e.g., 'dirt', 'snow', 'shadow').

    Returns:
    PIL.Image: The image with the applied effects.
    """

    width, height = image.size
    effect_types = effect_dict['plate']['effect']
    effect_strength = effect_dict['plate']['strength']
    headlights_mode = effect_dict['headlights']['mode']
    
    for effect_type in effect_types: 
        
        #Adding a dirt or snow surface effect (both created due the perlin noise function) 
        if effect_type == 'dirt' or effect_type == 'snow':
            
            #Set effect strength option based on config file input or select randomly
            effect_strength_selection = {"subtle": 0, "mild": 12.5, "moderate": 25, "strong": 37.5, "intense": 50} 
            if effect_setting == 'specified':
                effect_strength_value = effect_strength_selection[effect_strength]
            else:
                #Define effect strength options and select one randomly with different possibilities (subtle = 30%, mild = 25%, moderate = 20%, strong = 15%, intense = 10%)
                effect_strength_value = np.random.choice(list(effect_strength_selection.values()), p=[0.3,0.25,0.2,0.15,0.1])

            #Creating the noise value image
            noise_image = np.zeros((height, width), dtype=np.float32)
            
            #Selecting a randome value in the ranges of the parameters for the perlin noise function
            scale=800
            octaves = np.random.randint(7, 8)
            persistence = np.random.uniform(0.5, 0.7)
            lacunarity = np.random.uniform(2.0, 3.5)

            #Generate a perlin noise layer for each pixel in the image
            for y in range(height):
                for x in range(width):
                    nx = x / scale
                    ny = y / scale

                    #Calculate the noise values using the perlin noise function and the randomly selected parameters
                    noise_value = noise.pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=width, repeaty=height)
                    noise_image[y, x] = noise_value

            #Normalize the noise values to the range [0, 255]
            min_val = noise_image.min()
            max_val = noise_image.max()
            noise_image = (((noise_image - min_val) / (max_val - min_val) * 255)).astype(np.uint8)

            #Applying the additional strength effect to the noise values by reducing the pixel values
            dirt = Image.fromarray(noise_image, mode='L')
            dirt = dirt.point(lambda p: p - effect_strength_value)

            #Create a second layer
            dirt_rgba = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            #Apply the perlin noise values as the alpha channel to the new RGBA image
            dirt_rgba.putalpha(dirt)

            # Composite the dirt or snow effect onto the original image
            if effect_type == 'dirt':

                #Apply a dirt effect by combining the original image and a black image with the perlin noise mask based on the alpha values
                image = Image.composite(image, Image.new('RGBA', image.size, (0, 0, 0, 255)), dirt_rgba)
            elif effect_type == 'snow':

                #Apply a dirt effect by combining the original image and a black image with the perlin noise mask based on the alpha values
                image = Image.composite(image, Image.new('RGBA', image.size, (255, 255, 255, 255)), dirt_rgba)
            else:
                image = image


        elif effect_type == 'shadow':
            # Creates a shadow layer
            shadow_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow_layer)
            min_shadow = 0.2 #Minimum shadow coverage of the image height
            max_shadow = 0.8 #Maximum shadow coverage of the image height

            #Setting shadow strength due to transparenzy value
            effect_strength_selection = {"subtle": 150, "mild": 170, "moderate": 190, "strong": 210, "intense": 230} 
            if effect_setting == 'specified':
                alpha = effect_strength_selection[effect_strength]
            else:
                #Setting a randome transparency value for the shadow
                alpha = np.random.randint(150, 230)
            
            #Choosing 50/50 between horizontal and diagonal shadow
            if np.random.rand() > 0.5:
                #Horizontal shadow
                shadow_height = np.random.uniform(min_shadow, max_shadow) * height
                
                draw.rectangle([(0, 0), (width, shadow_height)], fill=(0, 0, 0, alpha))
            else:
                #polygon shadow
                start_from_left = np.random.rand() > 0.5
                if start_from_left:
                    #Define points for a diagonal shadow starting from the left
                    x1 = 0
                    y1 = 0
                    x2 = np.random.randint(width * 0.6, width)
                    y2 = y1
                    x3 = np.min([np.random.randint(width * 0.6, width), x2])
                    y3 = np.random.randint(height * min_shadow, height * max_shadow)
                    x4 = x1
                    y4 = y3

                else:
                    #Define points for a diagonal shadow starting from the right
                    x1 = width
                    y1 = 0
                    x2 = x1
                    y2 = np.random.randint(height * min_shadow, height * max_shadow)
                    x3 = np.random.randint(0, width * 0.4)
                    y3 = y2
                    x4 = np.min([np.random.randint(0, width * 0.4), x3])
                    y4 = y1

                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

                draw.polygon(points, fill=(0, 0, 0, alpha))
            
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(10))
            
            #Composite the shadow layer onto the original image
            image = Image.alpha_composite(image.convert('RGBA'), shadow_layer)

        else:
            image = image
        
    output = image

    return output