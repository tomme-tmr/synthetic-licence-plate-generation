# synthetic-licence-plate-generation

This project was created as part of a master's thesis by Lars G. and further developed by Tom-Marvin Rathmann, first in a project thesis and then in a second master's thesis.

The aim of the project is to generate synthetic data for the research field of automatic license plate recognition (ALPR). Therefor License plates on existing vehicle images are to be replaced by synthetic German license plates in order to be able to automatically generate training data for training purposes in the field of machine learning.
In the further developments, additional functionalities were implemented to artificially create synthetic weather artifacts in order to generate training data under difficult conditions.

The python script “full_process_bereinigt.py” combines all the functionalities and thus represents the image processing pipeline.
The python script “gpt_extraction.py” is used to extract features using an OpenAI model.
The python script “evaluate_gpt_results.py” is used to evaluate the quality of the license plate extraction using GPT.


Setup and run the pipeline:

1. open the folder in the terminal
cd '<folder path>'

2. create a virtual environment
python3 -m venv venvLP

3. activate the virtual environment
source venvLP/bin/activate

4. install the required packages
python -m pip install -r requirements.txt

5. run the image processing pipeline
python3 full_process_bereinigt.py
