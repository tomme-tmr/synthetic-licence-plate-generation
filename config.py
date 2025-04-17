'''
Created on Sun Dec 16 2024
@author: Tom-Marvin Rathmann

This module contains general configuration settings and directory structure definitions for the synthetic number plate creation project.
It provides essential variables such as the OpenAI API key and directory paths that are used across
the image processing pipeline. The configuration facilitates flexible adjustments of input and output
locations and allows control over key parameters, including the use of GPT for extracting the number plate.

Key Variables:
- OpenAI_api_key: Stores the OpenAI API key for accessing OpenAI services, such as GPT-based number plate extraction.
- DIR: A dictionary containing paths for various project files, such as configuration files, input and output files.
'''

import pathlib

#Variable to set the OpenAI Key
OpenAI_api_key = ""

#Set up the base data directory
data_dir = pathlib.Path() / "data"

#Dictionary for all the relevant paths, used in this project
DIR = {
    "config": data_dir / "config",
    "base": pathlib.Path(),
    "input": data_dir / "input",
    "output": data_dir / "output",
    "fahrzeugbilder": data_dir / "input" / "Fahrzeugbilder",
    "synthetic_pictures": data_dir / "output" / "synthetic_pictures"
}

#Create all directories if they do not exist
[dn.mkdir(parents=True, exist_ok=True) for dn in DIR.values()]