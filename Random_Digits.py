# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 22:08:01 2023
@author: LarsG
This function generates random letters and numbers for the synthetic numberplate.
The letters and numbers are assigned to their specific position on the numberplate.
Based on the chosen frequency, the vintage letter "H" and the electric letter "E" are added to the numberplate.
The function returns:
    - The list sorted_variables, which ioncludes the sequence for the synthetic numberplate.
    - The list values_list, which includes the letters and numbers of the sythetic numberplate (not sorted).
    - The number_digits, which shows the number of digits which are created for the recognition numbers
    - The composition_np, the length of the different blocks
    - The len_first_block, which includes especially the value for the length of the first block

"""
import string

def generate_digits ():
    import random

    text_numbers = ['E']+['H']

    #list with all big letters
    all_letters = string.ascii_uppercase
    #Remove letters "X" and "Y"
    letters_wanted = [letter for letter in all_letters if letter not in ['X', 'Y']]
    #Comment: There is no country state or district starting with X or Y in Germany. 
    #With the following code, it is possible to have X or Y as the first letter on the numberplate,
    #Because of the randomnes of letter 1, letter 2 or letter 3 for the beginning.
    
    text1 = random.choice(letters_wanted) #X and Y are not available in "real life"-> www.kba.de
    text2 = random.choice(string.ascii_uppercase)
    text3 = random.choice(string.ascii_uppercase)
    text4 = random.choice(string.ascii_uppercase)
    text5 = random.choice(string.ascii_uppercase)
    number1 = str(random.randint(1, 9))
    number2 = str(random.randint(0, 9)) 
    number3 = str(random.randint(0, 9))
    number4 = str(random.randint(0, 9))
    end_digit = str(random.choice(text_numbers))
    
    # Dictionary
    dictionary_np = {
        'letter1': text1,
        'letter2': text2,
        'letter3': text3,
        'letter4': text4,
        'letter5': text5,
        'number1': number1,
        'number2': number2,
        'number3': number3,
        'number4': number4,
        'xend_digit': end_digit
    }
    
    import random
    
    #Weighting of the number of letters based on the theoretical frequency
    weights_letter45 = [1, 26]  # 'letter 4' and 'letter 5'
    #Weighting of the number of digits for the recognition numbers based on the theoretical frequency
    weights_number = [9, 81, 720, 6500]  
    
    #Chose of the first selection for the country/district state: 'letter 1', 'letter 2', 'letter 3'
    first_block = random.sample(['letter1', 'letter2', 'letter3'], random.randint(1, 3))
    len_first_block = len(first_block)
    
    #Chose of the second selection for the recognition letters: 'letter 4' und 'letter 5'
    second_block = random.choices([['letter4'], ['letter5']], weights=weights_letter45)[0]
    if second_block == ['letter5']:
        first_block.append('letter4')
    
    #Calculation how many letters/numbers are available for the third selection 
    #Third selection is the recognition numbers. 
    #The maximum number of digits on a numberplate is 8
    remaining_variables = 8 - len(first_block) - (2 if second_block == 'letter5' else 1)
    len_second_block = 8 - remaining_variables - len_first_block
    
    ##Chose of the third selction for recognition numbers: 'number 1', 'number 2', 'number 3', 'number 4'
    third_block_numbers = ['number1', 'number2', 'number3', 'number4']
    max_vars_third_block = min(remaining_variables,4)  # Maximale Anzahl von Variablen im dritten Block
    num_number_vars = random.choices(range(1, max_vars_third_block +1), weights=weights_number[:max_vars_third_block])[0]
    length_third_block = random.sample(third_block_numbers, num_number_vars)
    length_3rd_block = len(length_third_block)

    
    #Add/replace the end digit by a "H" or "E" based on the given frequency
    # State 23/10/01 1.3 million electric vehicles ("E"), 800.000 historical cars ("H") and 60 millions non electric vehicles
    electricity_vintage = 0
    if random.randint(1, 60) == 1:  
        length_third_block[random.randint(0, len(length_third_block) - 1)] = 'xend_digit'
        electricity_vintage = 1
        length_3rd_block = length_3rd_block-1
    elif random.randint(1, 60) == 1 and len(first_block) + len(second_block) + len(length_third_block) < 8 and "xend_digit" not in length_third_block:
        length_third_block.append('xend_digit')
        electricity_vintage = 1

    
    #Combine all blocks to the final sequence for the numberplate
    selected_variables = first_block + second_block + length_third_block
    
    ############## Final variabel ###############
    sorted_variables = sorted(selected_variables)
    number_digits = len(sorted_variables)
    #print(sorted_variables)
    
    #Create a list which is returned by the function
    composition_np = {
        'district': len_first_block,
        'recognition_letter': len_second_block,
        'number': length_3rd_block,
        'electricity_vintage': electricity_vintage
    }
    
    #print(sorted_variables)
    # print(composition_np)
    # print(number_digits)
    #print("Ausgewählte Variablen:", selected_variables)
    #print("anzahl Variablen", len(selected_variables))
    
    #########################################################################################
    
    
    #Create an additional list which is returned by the function.
    values_list = []
    
    #Access to the dictionary entries by the sorted variables and writing into the list.
    for variable in sorted_variables:
        if variable in dictionary_np:
            value = dictionary_np[variable]
            values_list.append(value)
            
    return(sorted_variables,values_list, number_digits, composition_np, len_first_block )
    
    #print("Liste der Werte:", values_list)
    
#composition = generate_digits()[1]
