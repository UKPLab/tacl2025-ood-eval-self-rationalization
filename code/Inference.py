import os
import wandb
os.environ["WANDB_WATCH"]='false'
os.environ["WANDB_LOG_MODEL"]='false'
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "f943c9ab18325b5cd45241ad2df308268a65c7b8"
os.environ["WANDB_CONFIG_DIR"] = "/ukp-storage-1/jyang/wandb/.config"
os.environ["WANDB_CACHE_DIR"] = "/ukp-storage-1/jyang/wandb/.cache"
os.environ["WANDB_JOB_TYPE"] = 'test'

from tqdm import tqdm
from datetime import datetime
import random 
import pandas as pd 
import numpy as np
from metrics import evaluate
from nli_demo import evaluate_score
from load_target_dataset import load_format_data
from utils import sep_label_explanation

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets
import git
import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    set_seed
)

CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}

label2text = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class MyDataset(Dataset):
    def __init__(self, df, tokenizer):

        input_data = tokenizer.batch_encode_plus(df['input_string'].to_list(), 
#                                            max_length = 200, 
                                           return_tensors="pt", 
                                           padding=True, 
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                          )
        
        dec = tokenizer.batch_encode_plus(
                                    df['answer_string'].to_list(),
#                                     max_length=200,
                                    return_tensors="pt", 
                                    padding=True,
                                    return_token_type_ids=False,
                                    return_attention_mask=True,
                                )
        
        self.input_ids = input_data['input_ids']
        self.attention_mask =input_data['attention_mask']

        
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
    
    def __len__(self):
        return len(self.input_ids)

def inference(model, tokenizer, seed, data, test_bsz, result_path, explanation_sep):

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    df_data = data.to_pandas()
    data_set = MyDataset(df_data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=test_bsz, shuffle=False, num_workers = 2)

    explanations = []
    labels = []
    label_probabilities = []
    answers_explanations = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_mask.to(device)
        output = model.generate(input_ids = input_ids, 
                                attention_mask=attention_masks, 
                                output_scores=True, 
                                max_length=200, 
                                return_dict_in_generate=True)
        
        answer_explanation = tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True)
        answers_explanations+=answer_explanation
        _, exp = sep_label_explanation(answer_explanation, explanation_sep)
        # if exp=='' or exp== np.nan:
        #     exp=explanation_sep
        explanations += exp
        probabilities = F.softmax(output.scores[0], dim=1)
        token_probability = probabilities[:,[3,7163,27252]] # token id for "en", "neutral" and "contradiction"
        label_probabilities+=token_probability.cpu().tolist()
        labels+=token_probability.argmax(dim=-1).cpu().tolist()
        # answers_gold.append(model.generate(input_ids = input_ids, 
        #                                    attention_mask=attention_masks, 
        #                                    max_length=200,
        #                                    pad_token_id=tokenizer.pad_token_id,
        #                                    eos_token_id=tokenizer.eos_token_id).cpu().tolist())
        torch.cuda.empty_cache()


    generations_list = []
    for i in range(len(labels)):
        ans_exp = answers_explanations[i].replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
        # label = label2text[labels[i]]
        # answer_explanation = explanation_sep.join((label, explanation))
        generations_list.append(ans_exp)
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '/test_generation_batch.txt', 'w') as f:
        for line in generations_list:
            f.write(f"{line}\n")
    
    return labels, explanations, label_probabilities
                  

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='few-shot-inference', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default=None, type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, 
                       help="subset dataset")
    parser.add_argument("--data_split", default='test', type=str, required=False, 
                       help="dataset split: train, val, test")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default='t5-large', type=str, required=False, 
                       help="model base")
    parser.add_argument("--n_shots", default= 8, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--explanation_source", default= 'na', type=str, required=False, 
                       help="source of explanation") 
    parser.add_argument("--seed", default= 42, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--sample_seed", default= 0, type=int, required=False, 
                       help="sample selection seed")
    parser.add_argument("--test_bsz", default= 64, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--io_format", default= 'standard', type=str, required=False, 
                       help="nli prompt format")
    parser.add_argument("--source_select", action='store_true',
                       help="selecting samples for training source dataset selection")
    parser.add_argument("--model_path", default= '../model', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--result_path", default= '../result', type=str, required=False, 
                       help="path to save model")
    
    args = parser.parse_args()
    
    relative_path = "/".join((args.source_dataset_name, args.sample_selection, 'sub'+str(args.data_sub), 'nt'+str(args.n_shots)))
    # if args.sample_selection == 'random':
    #     relative_path = "+".join((relative_path, 'ssd'+str(args.sample_seed)))
    
    if args.n_shots == 0:     
        run_name = "/".join(('zero_shot',args.target_dataset_name))
        result_path = '/'.join((args.result_path, run_name))
    else:    
        model_path = '/'.join((args.model_path, relative_path))
        run_name = "/".join((args.target_dataset_name, relative_path))
        result_path = '/'.join((args.result_path, run_name))

    # if args.explanation_sep != 'na':
    #     run_name = "/".join((args.target_dataset_name, relative_path))
    #     model_path = '/'.join((args.model_path, relative_path))
    #     result_path = '/'.join((args.result_path, run_name))

    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.target_dataset_name, args.sample_selection],
           group=args.target_dataset_name,
           config = args,
           save_code = True)    

    set_seed(args.seed)
    set_other_seeds(args.seed)

    print("Loading data...")
    data = load_format_data(args.target_dataset_name, args.data_split)
    ## Sample for training source dataset selection
    if args.source_select:
        data = data.shuffle(seed=42)
        if len(data)>500:
            data=data.select(range(500))  
        
        run_name = "/".join((args.target_dataset_name, 'best_model'))
        result_path = '/'.join((args.result_path, args.target_dataset_name, 'best_model'))
        
    print("Loading model...")
    
    if args.n_shots == 0:
        tokenizer_name = TOKENIZER_MAPPING[args.model_class]
        tokenizer = tokenizer_name.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        tokenizer_name = TOKENIZER_MAPPING[args.model_class]
        tokenizer = tokenizer_name.from_pretrained(model_path, local_files_only=False)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=False)

    print("Model inferencing...")
    labels, explanations, _ = inference(
        model=model, 
        tokenizer=tokenizer,
        seed=args.seed, 
        data=data, 
        test_bsz=args.test_bsz, 
        result_path=result_path,
        explanation_sep=args.explanation_sep
    )
    
    print("Evaluating results...")
    # evaluate classification accuracy
    results, cm = evaluate(
        result_path,
        data,
        tokenizer,
        "test",
        task=args.target_dataset_name,
        labels=labels,
        explanations=explanations
    )
    
    # evaluate explanation acceptability
    df_data = data.to_pandas()
    if args.n_shots!=0:
        exp_score = evaluate_score(result_path, df_data, args.target_dataset_name, args.test_bsz//2)
        results['accept_score'] = exp_score
    print(results)

    df_cm = pd.DataFrame(cm)

    wandb.log({"confusion_matrix": wandb.Table(dataframe=df_cm)})

    wandb.log(results)
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()

if __name__ == "__main__":
    main() 