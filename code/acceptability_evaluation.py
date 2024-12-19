import os
import wandb
os.environ["WANDB_SILENT"]="true"

import pandas as pd
import numpy as np
import json
import datasets
import torch
import re
#from utils import sep_label_explanation
import argparse
from nli_demo import get_scores
from load_custom_dataset import load_raw_data, load_format_data

label_mapping = {
    'esnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'efever': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'joci': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'conj': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'sick': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mpe': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'glue_diagnostics': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "wnli": {0: 'contradiction', 1: 'entailment'},
    "add_one": {0: 'contradiction', 1: 'entailment'},
    "dnc": {0: 'contradiction', 1: 'entailment'},
    "hans": {0: 'entailment', 1: 'contradiction'}, # the label is opposite of the others
    "fm2": {0: 'entailment', 2: 'contradiction'},
    "covid_fact": {0: 'entailment', 2: 'contradiction'},
    'vitaminc':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'snopes_stance':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'scifact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-combined':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'factcc':{0: 'entailment', 1: 'contradiction'},
    'qags_cnndm':{0: 'entailment', 1: 'contradiction'},
    'qags_xsum':{0: 'entailment', 1: 'contradiction'},
    'xsum_hallucination':{0: 'entailment', 1: 'contradiction'},
}

def df_formatting(row):

    input_string='premise: '+ row['premise'] + ' hypothesis: ' + row['hypothesis'] + ' answer: '+ row['label'] + ' explanation: ' + row['explanation']
    # answer_string=f'{expl_gen} '

    return pd.Series([input_string])

def extract_data_from_file(file_path):
    # Reading the content of the provided file
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

    with open(file_path, 'r') as file:
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
    
def extract_relationship_explanation(input_string, model_type='olmo'):
    if model_type=='olmo':
        
        # Regular expression patterns to extract relationship and explanation
        relationship_pattern = r'"relationship"\s*:\s*"([^"]+)"'
        explanation_pattern = r'"explanation"\s*:\s*"(.*?)"\s*\}'
        
        # Search for the patterns in the input string
        relationship_match = re.search(relationship_pattern, input_string)
        explanation_match = re.search(explanation_pattern, input_string)
        
        # Extract the values if the patterns are found
        relationship = relationship_match.group(1) if relationship_match else "none"
        explanation = explanation_match.group(1) if explanation_match else 'none'
    if model_type=='t5':
        explanation_sep=' "explanation: " '
        line_split = input_string.split(explanation_sep)
        if len(line_split) > 1:
            relationship=line_split[0].strip() #text label: entailment, neutral, contradiction
            explanation=line_split[1].strip()
        else:
            try:
                # print(f"This line couldn't be processed (most likely due to format issue): {line}")
                relationship=input_string.split()[0] #text label: maybe nonsense, we assume the first word is always the label
                explanation=' '.join(input_string.split()[1:])
            except:
                relationship=input_string #the line is totally empty
                explanation=''  
    return relationship, explanation

def merge_answers(label, num_classes):
    if num_classes==2: # convert predicted label from 3 classes to 2 classes
        label = "contradiction" if label.lower() in ["neutral", "contradiction"] else "entailment" 
    return label

def read_from_path(df_result,path,dataset_name,model_type='t5'):
    with open(path+'/test_generation_batch.txt', 'r') as file:
        lines = file.readlines()
    #     print(lines)  # This will print a list where each element is a line from the file
    df_result['ans_exp'] = lines
    answers,rationales = zip(*[extract_relationship_explanation(s,model_type=model_type) for s in df_result['ans_exp']])
    df_result['label']=[label_mapping[dataset_name][l] for l in df_result['label']]
    num_classes=len(set(df_result['label']))
    answers=[merge_answers(a,num_classes) for a in answers]
    # gold_label=[merge_answers(a, num_classes) for a in df_result['label']]

    df_result['explanation']=rationales
    df_result['predicted_label']=answers
    df_result[['input_string']]=df_result.apply(df_formatting,axis=1)

    return df_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='auto-j-eval', type=str, required=False, help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, help="training dataset name")
    parser.add_argument("--target_dataset_name", default='mpe', type=str, required=False, help="target dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, help="subset dataset")
    parser.add_argument("--n_shots", default= 32, type=int, required=False, help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, help="sample selection method")
    parser.add_argument("--model_type", default= 't5', type=str, required=False, help="model type")
    parser.add_argument("--result_path", default= 'result-slurm', type=str, required=False, help="result path")
    parser.add_argument("--batch_size",default=16, type=int, help="evaluation batch size")
    parser.add_argument("--fix", action='store_true', help="alternative t5")
    parser.add_argument("--specified_path", action='store_true', help="alternative olmo")
    args = parser.parse_args()
    
    # save_path='/home/few-shot-fact-checking/'+args.result_path+args.target_dataset_name+'/'+args.source_dataset_name+'/'+args.sample_selection+'/sub'+str(args.data_sub)+'/nt'+str(args.n_shots)+'/'
    if args.model_type=='t5':
        if args.n_shots==5000 and args.source_dataset_name=='efever':
            save_path='/'.join(('/home/few-shot-fact-checking/result-slurm',args.target_dataset_name,args.source_dataset_name,'full-shot','sub0','nt'+str(args.n_shots)))
        elif args.sample_selection in ['random', 'ambiguous', 'accept-ambiguous'] and args.source_dataset_name=='esnli':
            save_path='/'.join(('/home/few-shot-fact-checking/result',args.target_dataset_name,args.source_dataset_name, args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
        else:
            save_path='/'.join(('/home/few-shot-fact-checking/result-slurm',args.target_dataset_name,args.source_dataset_name,args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
            
    elif args.model_type=='olmo':
        if args.specified_path:
            save_path='/'.join(('/home/few-shot-fact-checking/result',args.target_dataset_name,'allenai-OLMo-1.7-7B-hf', args.source_dataset_name,args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
        else:
            save_path='/'.join(('/home/few-shot-fact-checking/result-28',args.target_dataset_name,'allenai-OLMo-1.7-7B-hf', args.source_dataset_name,args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
            
    wandb.init(project=args.project_name,
       name=save_path,
       tags=[args.target_dataset_name],
       group=args.source_dataset_name,
       config = args,
       save_code = True)  

    model_type = "11b" # or the local directory to store the downloaded model

    data_test = load_raw_data(args.target_dataset_name, 'test')
    df_result=data_test.to_pandas()
    
    if args.fix:
        if os.path.exists(save_path+"/test_posthoc_analysis_1.txt"):
            filename = save_path+"/test_posthoc_analysis_1.txt"
        else:
            filename = save_path+"/test_posthoc_analysis.txt" 
        df = extract_data_from_file(filename)

        df_result['label']=[label_mapping[args.target_dataset_name][l] for l in df_result['label']]
        df_result['explanation'] = [row['Predicted'].split('|')[1] for _, row in df.iterrows()]
        df_result['input_string'] = ['premise: '+ row['premise'] + ' hypothesis: ' + row['hypothesis'] + ' answer: '+ row['label'] + ' explanation: ' + row['explanation'] for _, row in df_result.iterrows()]   
        
    else:
        df_result=read_from_path(df_result,save_path,args.target_dataset_name,model_type=args.model_type)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scores = get_scores(
    df_result['input_string'].to_list(),
    model_type,
    batch_size=args.batch_size,
    device=device,
    verbose=False)

    df_result['accept_scores_11b']=scores
    df_result.to_json(save_path+'/exp_scores.json', orient='records', lines=True)
    print(np.mean(scores))

    wandb.log({"accept_score_11b": np.mean(scores)})
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()


if __name__ == "__main__":
    main() 