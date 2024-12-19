import sys
import os
sys.path.append("/few-shot/auto-j/codes/usage/")
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
from load_custom_dataset import load_raw_data, load_format_data
from vllm import LLM, SamplingParams
from constants_prompt import build_autoj_input # constants_prompt -> codes/constants_prompt.py
label_mapping = {
    'joci': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'conj': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'sick': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mpe': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'glue_diagnostics': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "wnli": {0: 'not entailment', 1: 'entailment'},
    "add_one": {0: 'not entailment', 1: 'entailment'},
    "dnc": {0: 'not entailment', 1: 'entailment'},
    "hans": {0: 'entailment', 1: 'not entailment'}, # the label is opposite of the others
    "fm2": {0: 'entailment', 2: 'contradiction'},
    "covid_fact": {0: 'entailment', 2: 'contradiction'},
    'vitaminc':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'snopes_stance':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'scifact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-combined':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'factcc':{0: 'entailment', 1: 'not entailment'},
    'qags_cnndm':{0: 'entailment', 1: 'not entailment'},
    'qags_xsum':{0: 'entailment', 1: 'not entailment'},
    'xsum_hallucination':{0: 'entailment', 1: 'not entailment'},
}

def df_formatting(row):

    premise = row["premise"]
    hypothesis = row["hypothesis"]
    answer_gold = row["label"]
    # expl_gold = row["explanation_1"]
    # expl_gen= row['explanation']

    input_string=f"Based the premise \"{premise}\", please explain why the hypothesis \"{hypothesis}\" is {answer_gold}."
    # answer_string=f'{expl_gen} '

    return pd.Series([input_string])

def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            pred_label = 0
        elif pred_rest.startswith('response 2'):
            pred_label = 1
        elif pred_rest.startswith('tie'):
            pred_label = 2
    return pred_label


def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        marker = "Rating: [["
        start = score_output.find(marker)
        if start != -1:
            start += len(marker)
            end = score_output.find("]]", start)
            if end != -1:
                return float(score_output[start:end].strip())
            else:
                return 0.0
        else:
            return 0.0
    else:
        return 0.0

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
        label = "not entailment" if label.lower() in ["neutral", "contradiction"] else "entailment" 
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
    parser.add_argument("--max_seq_length",default=512, type=int, help="max sequence length for model and packing of the dataset")

    args = parser.parse_args()
    
    # save_path='/home/few-shot-fact-checking/'+args.result_path+args.target_dataset_name+'/'+args.source_dataset_name+'/'+args.sample_selection+'/sub'+str(args.data_sub)+'/nt'+str(args.n_shots)+'/'
    if args.model_type=='t5':
        if args.n_shots==5000 and args.source_dataset_name=='efever':
            save_path='/'.join(('/home/few-shot-fact-checking/result-slurm',args.target_dataset_name,args.source_dataset_name,'full-shot','sub0','nt'+str(args.n_shots)))
        elif args.sample_selection=='random':
            save_path='/'.join(('/home/few-shot-fact-checking/result',args.target_dataset_name,args.source_dataset_name,'random','sub'+str(args.data_sub),'nt'+str(args.n_shots)))
        else:
            save_path='/'.join(('/home/few-shot-fact-checking/result-slurm',args.target_dataset_name,args.source_dataset_name,args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
            
    elif args.model_type=='olmo':
        if args.n_shots==5000:
            save_path='/'.join(('/home/few-shot-fact-checking/result-28',args.target_dataset_name,'allenai-OLMo-1.7-7B-hf', args.source_dataset_name,'nt5000'))
        else:
            save_path='/'.join(('/home/few-shot-fact-checking/result-28',args.target_dataset_name,'allenai-OLMo-1.7-7B-hf', args.source_dataset_name,args.sample_selection,'sub'+str(args.data_sub),'nt'+str(args.n_shots)))
    wandb.init(project=args.project_name, 
       name=save_path,
       tags=[args.target_dataset_name],
       group=args.source_dataset_name,
       config = args,
       save_code = True)  

    num_gpus = torch.cuda.device_count()
    model_name_or_dir = "GAIR/autoj-13b" # or the local directory to store the downloaded model
    llm = LLM(model=model_name_or_dir,tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    
    data_test = load_raw_data(args.target_dataset_name, 'test')
    df_result=data_test.to_pandas()
    df_result=read_from_path(df_result,save_path,args.target_dataset_name,model_type=args.model_type)
        
    input_gen=[]
    for index, row in df_result.iterrows():
        input_gen.append(build_autoj_input(prompt=row['input_string'],
                                     resp1=row['explanation'],
                                     resp2=None, protocol="single"))  # for single response evaluation
    
    outputs_gen = llm.generate(input_gen, sampling_params)
    df_result['rate_gen'] = [item.outputs[0].text for item in outputs_gen]
    df_result.to_json(save_path+'/auto_j_single_gen.json', orient='records', lines=True)
    scores = [extract_single_rating(judg) for judg in df_result['rate_gen']] # `extract_single_rating` for single-response evaluation 
    print(np.mean(scores))

    wandb.log({"auto_j_score": np.mean(scores)})
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()


if __name__ == "__main__":
    main() 