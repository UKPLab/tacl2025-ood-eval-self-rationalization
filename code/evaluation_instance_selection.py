import random
import pandas as pd
import os
# from nli_demo import evaluate_score
import re

# Function to correctly parse the file and extract the relevant information with adjustments for missing sections
def parse_file_properly(file_path):
    # Initialize the dictionary to hold our data
    if os.path.exists(file_path+"/test_posthoc_analysis_1.txt"):
        filename = file_path+"/test_posthoc_analysis_1.txt"
    else:
        filename = file_path+"/test_posthoc_analysis.txt"  
        
    data = {
        "Hypothesis": [],
        "Premise": [],
        "Correct": [],
        "Predicted": [],
        "Considered Correct": []
    }

    # Initialize a temporary dictionary to hold the current set of data
    temp_data = {key: "" for key in data}

    # Regular expression pattern to match the labels
    pattern = re.compile(r"^(Hypothesis|Premise|Correct|Predicted|Considered Correct):")

    with open(filename, 'r') as file:
        for line in file:
            # Check if the line starts with one of our labels
            match = pattern.match(line)
            if match:
                # If we have collected text for a previous label, add it to the corresponding list
                current_label = match.group(1)
                if temp_data[current_label]:  # If there's already text for this label, it means we've reached a new record
                    for label, text in temp_data.items():
                        data[label].append(text)
                    temp_data = {key: "" for key in data}  # Reset for the next record
                # Start collecting text for the new label
                temp_data[current_label] = line[len(match.group(0)):].strip()
            elif current_label:
                # This is a continuation of the text for the current label
                temp_data[current_label] += " " + line.strip()
        
        # Don't forget to save the last record after the loop ends
        for label, text in temp_data.items():
            data[label].append(text)

    # Remove the 'Correct' key if it has only empty strings, as we didn't find this label in the unique lines
    if all(len(text.strip()) == 0 for text in data["Correct"]):
        data.pop("Correct")

    # Return the data with potential empty strings where sections were missing
    df_data = pd.DataFrame(data)
    
    return df_data

def select_from_dir(path,lent):
    # Read the file and group by attributes
#     with open(path+"/test_posthoc_analysis.txt", 'r') as file:
#         lines = file.readlines()

#     grouped_data = group_lines_by_attribute(lines)
    # Convert the grouped data into a pandas DataFrame
    df = parse_file_properly(path)
#     print(df)
    df = df.sample(frac=1,random_state=2).reset_index(drop=True)
    df = df.loc[df['Considered Correct'] == 'True ']

    df = df.rename(columns={"Hypothesis": "hypothesis", "Premise": "premise"})
    # labels, explanations = [l, x for _, row in df.iterrows()]
    df['label'] = [row['Predicted'].split('|')[0] for _, row in df.iterrows()]
    df['explanation'] = [row['Predicted'].split('|')[1] for _, row in df.iterrows()]

    selected_df = pd.DataFrame()
    for l in set(df['label']):
        tmp_df = df.loc[df['label']==l][:lent]
        tmp_df['dataset'] = [path.split('/')[2]]*lent
        tmp_df['sample_selection'] = [path.split('/')[4]]*lent
        tmp_df['n_shots'] = [path.split('/')[6]]*lent
        tmp_df['source_dataset']=[path.split('/')[3]]*lent
        selected_df = pd.concat([selected_df, tmp_df], ignore_index=True)

    return selected_df

source_dataset = ['esnli','efever']
sample_selection = ['fastvotek','accept-fastvotek']
target_datasets = ['sick','vitaminc','xsum_hallucination']
n_shots=64
selected_sub=0

dirs = []
for sd in source_dataset:
    for ss in sample_selection:
        if sd=='efever' and ss!='accept-fastvotek':
            continue
        for td in target_datasets:
            dirs.append('/'.join(('../result',td,sd,ss,'sub'+str(selected_sub),'nt'+str(n_shots))))

lent=15
df = select_from_dir(dirs[0],lent)
for path in dirs[1:]:
    df=pd.concat([df,select_from_dir(path,lent)])

df = df.drop(columns=['Correct', 'Predicted', 'Considered Correct'])
df['label'] = df['label'].replace(['not_entailment ','not_entailment'], 'contradiction')
df.to_json('../select_best_few_shots_instances.json', orient='records', lines=True)