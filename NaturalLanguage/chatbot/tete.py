import numpy as np
import pandas as pd

'''crop_dialogue_df = pd.read_csv('data/crop_dialogue.csv', sep=';')
df_desc = pd.read_csv('RandDescrip.csv', sep=';')

crop_dialogue_df = crop_dialogue_df._append(df_desc, ignore_index=True)

crop_dialogue_df.to_csv('data/crop_dialogue.csv', sep=';', index=False)

print(df_desc)'''


import yaml
import os

pairs = []

directory = 'chatterbot_corpus/data'

# Traverse the directory and its subdirectories
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a .yml file
        if file.endswith('.yml'):
            # Construct the full path to the .yml file
            filepath = os.path.join(root, file)

            # Load the .yml file with utf-8 encoding
            with open(filepath, 'r', encoding='utf-8') as yamlfile:
                data = yaml.safe_load(yamlfile)
                # Iterate over all keys in the YAML data
                for key in data.keys():
                    # Check if the key contains conversation pairs
                    if isinstance(data[key], list) and len(data[key]) > 0 and isinstance(data[key][0], list) and len(
                            data[key][0]) == 2:
                        for conversation_pair in data[key]:
                            input_pattern = conversation_pair[0]
                            response = conversation_pair[1]
                            pairs.append((input_pattern, response))


data = {'conversations': pairs}

# Define the path to the new .yml file
output_file = 'output_pairs.yml'

# Write the pairs to the new .yml file
with open(output_file, 'w', encoding='utf-8') as yamlfile:
    yaml.safe_dump(data, yamlfile, allow_unicode=True)

print(pairs)
