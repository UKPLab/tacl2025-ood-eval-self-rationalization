import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

import wandb
os.environ["WANDB_WATCH"]='false'
os.environ["WANDB_LOG_MODEL"]='false'

# os.environ["WANDB_API_KEY"] = 'f943c9ab18325b5cd45241ad2df308268a65c7b8'
# os.environ["WANDB_MODE"] = "offline"

# import gpt3
# import logging
import argparse
import math
import json
from typing import List, Dict, Any, NewType
import numpy as np
from random import shuffle
import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
#     HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvaluationStrategy
from transformers.integrations import WandbCallback
import transformers
from transformers import Trainer

from feature_conversion_methods import format_instance, esnli_formatting, wt5_esnli_label_mapping, unified_qa_esnli_label_mapping
from sklearn.metrics import classification_report

# from custom_args import (
#     DataTrainingArguments,
#     ModelArguments
# )

from metrics import evaluate
from nli_demo import evaluate_score 
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
import git
import time
from datetime import datetime
import sys
from tqdm import trange
import random 
import pandas as pd 
import jsonlines
from copy import deepcopy 

InputDataClass = NewType("InputDataClass", Any)

CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}

esnli_label_mapping = {'SUPPORTS': 'entailment.', 'NOT ENOUGH INFO': 'neutral.', 'REFUTES': 'contradiction.'} 

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class SequenceCollator:
    def __init__(self, model, pad_token):
        self.model = model
        self.pad_token_mapping = {
            "labels": -100,
            "attention_mask": 0,
            "decoder_attention_mask": 0,
            "input_ids": pad_token,
        }

        self.columns = [
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_attention_mask",
        ]

    def __call__(self, examples: List[Dict[str, InputDataClass]]) -> Dict[str, torch.Tensor]:
        # re-format inputs for training
        batch = {}
        for key in examples[0].keys():
            if key in self.columns:
                tmp_list = []
                for item in examples:
                    tmp_list.append(item[key])

                # pad lists to max length
                if isinstance(tmp_list[0], list):
                    max_length = max(map(len, tmp_list))
                    tmp_list = [
                        el + [self.pad_token_mapping[key]] * (max_length - len(el))
                        for el in tmp_list
                    ]

                batch[key] = torch.tensor(tmp_list, dtype=torch.long)
        return batch
    
    
def relabel(row):
    if row['label'] == "NOT ENOUGH INFO" :
        return 1
    if row['label'] == "SUPPORTS" :
        return 0
    if row['label'] == 'REFUTES' :
        return 2
    
    
class MyDataset(Dataset):
    def __init__(self, df, tokenizer):

#         tokenizer_name = TOKENIZER_MAPPING[args.model_class]
#         tokenizer = tokenizer_name.from_pretrained(args.model_name)

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
#         self.label = torch.tensor(item['label'], dtype = torch.long)
#         self.label = dec["input_ids"]
#         self.decoder_attention_mask = dec['attention_mask']
#         self.question_encoding =input_data['input_ids']

        
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
#     self.label[index], self.decoder_attention_mask[index], self.question_encoding[index]
    
    def __len__(self):
        return len(self.input_ids)

    
def main():

    parser = argparse.ArgumentParser()
#     parser.add_argument("--run_name", default=None, type=str, required=True, 
#                        help="run name for the experiment")
    parser.add_argument("--project_name", default='few-shot', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--dataset_name", default='fm2', type=str, required=False, 
                       help="dataset name")
#     parser.add_argument("--dataset_file", default='/few-shot/qafact/data/fm2/processed_test.json', type=str, required=False, 
#                        help="dataset file")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default= None, type=str, required=True, 
                       help="model and tokenizer name")
    parser.add_argument("--seed", default= 479, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--sample_seed", default= 479, type=int, required=False, 
                       help="sample selection seed")
    parser.add_argument("--lr", default= 3e-5, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--train_bsz", default= 4, type=int, required=False, 
                       help="training batch size")
    parser.add_argument("--eval_bsz", default= 64, type=int, required=False, 
                       help="evaluation batch size")
    parser.add_argument("--test_bsz", default= 128, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--n_shots", default= 8, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--fewshot_eval_size", default= 350, type=int, required=False, 
                       help="number of samples for evaluation")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="input data path")
    parser.add_argument("--criteria", default= 'hypothesis', type=str, required=False, 
                       help="input data path")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--io_format", default= 'standard', type=str, required=False, 
                       help="nli prompt format")
    parser.add_argument("--num_classes", default= 2, type=int, required=False, 
                       help="number of classes")
    parser.add_argument("--save_model_path", default= '../outputs/', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--max_steps", default= 300, type=int, required=False, 
                       help="maximum number of steps")
    parser.add_argument("--eval_steps", default= 300, type=int, required=False, 
                       help="number of steps for evaluation")
    parser.add_argument("--rationale_eval", default= False, type=bool, required=False, 
                       help="whether to evaluate explanation") 
    
    args = parser.parse_args()
    
    run_folder = "/".join((args.model_name, args.source_dataset_name, args.dataset_name, args.sample_selection))
    run_name = "+".join((run_folder, 'st'+str(args.n_shots), args.explanation_sep, args.io_format, 'sd'+str(args.seed), 'ssd'+str(args.sample_seed)))
    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.sample_selection, args.dataset_name],
           group=args.dataset_name,
           config = args,
           save_code = True)    

#     sd = 479
    set_seed(args.seed)
    set_other_seeds(args.seed)
    task_name = 'esnli' ## our task is always NLI
    tokenizer_name = TOKENIZER_MAPPING[args.model_class]
    tokenizer = tokenizer_name.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    ##############################
    # TRAINNING SAMPLE SELECTION #
    ##############################
    data_splits = {'train': None, 'validation': None, 'test': None}
    original_data_splits = {'train': None, 'validation': None, 'test': None}  
    dataset = datasets.load_dataset('esnli')
    
    if args.source_dataset_name == 'mix':
        if args.sample_selection == 'fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/mix_fastvotek_filter.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)  

    if args.source_dataset_name == 'e-fever':
        if args.sample_selection == 'random':
            df_train = pd.read_json('efever_random_sampling_train.json', lines=True)
#             df_train = pd.read_json('e-fever-8-shots.json', lines=True)
            df_train = df_train.loc[df_train['explanation_1'] != 'empty']
            df_train['explanation_2'] = ''
            df_train['explanation_3'] = ''
            df_train['label'] = df_train.apply(relabel, axis=1)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
            
        if args.sample_selection == 'ms':
            df_train = pd.read_json('e-fever-8-shots.json', lines=True)
            df_train = df_train.loc[df_train['explanation_1'] != 'empty']
            df_train['explanation_2'] = ''
            df_train['explanation_3'] = ''
            df_train['label'] = df_train.apply(relabel, axis=1)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
            
        if args.sample_selection == 'fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/e-fever_fastvotek_filter_'+str(args.n_shots)+'.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)  
        
#         if args.sample_selection == 'e-fastvotek':
#             df_train = pd.read_json('/few-shot/icl-selective-annotation/e-fever_fastvotek_filter.json', lines=True)
#             pa_tab= pa.Table.from_pandas(df_train)
#             dataset['train'] = datasets.Dataset(pa_tab)  
            
        if args.sample_selection == 'nli-fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/e-fever_fastvotek_nli.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)  
            
        if args.sample_selection == 'acceptability':
            df_train = pd.read_json('/few-shot/prompt-engineering/few_shot_explanations/e-fever_accept_filter_'+str(args.n_shots)+'.json', lines=True)
            df_train['explanation_1'] = df_train['explanation']
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab) 
        
        if args.sample_selection == 'accept-fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/e-fever_accept_fastvotek_'+str(args.n_shots)+'.json', lines=True)
            df_train['explanation_1'] = df_train['explanation']
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)     
            
    if args.source_dataset_name == 'fever':
        if args.sample_selection == 'mr':
            df_train = pd.read_json('fever_random_sampling_train.json', lines=True)
    #         df_train = pd.read_json('e-fever-8-shots.json', lines=True)
    #         df_train = pd.read_json('fever-to-german.json', lines=True)
            df_train = df_train.loc[df_train['explanation_1'] != 'empty']
            df_train['explanation_2'] = ''
            df_train['explanation_3'] = ''
            df_train['label'] = df_train.apply(relabel, axis=1)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
        
        if args.sample_selection == 'acceptability':
            df_train = pd.read_json('/few-shot/prompt-engineering/few_shot_explanations/mr-fever_accept_filter_'+str(args.n_shots)+'.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab) 
        
        if args.sample_selection == 'accept-fastvotek':
            df_train = pd.read_json('/few-shot/prompt-engineering/few_shot_explanations/mr-fever_accept_filter_'+str(args.n_shots)+'.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab) 
            
        if args.sample_selection == 'nli-fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/fever_fastvotek_nli.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
#     if args.sample_selection == 'random': as default, we always select validation sets according to a random seed
    if args.source_dataset_name == 'esnli':

        if args.sample_selection == 'ms' or args.sample_selection == 'mr' : ## Manully selected training samples, so far only works with 8 shots
            with open('sridx.txt') as f:
                sample_idx = [int(line.rstrip()) for line in f]
            sample_select = dataset['train'].shuffle(seed = 479) ## this is a hack as we have already selected samples using this seed
            samples = sample_select[sample_idx]
            df_train = pd.DataFrame(samples)

            if args.sample_selection == 'mr':  ## also hacking and fixed
                ## Entail
                df_train['explanation_1'][0] = 'The premise states that a girl is with a red scarf wrapped around her neck, entailing the hypothesis that the girl is accessorized.'
                df_train['explanation_1'][1] = 'The hypothesis says the girl is jumping on a trampoline. Based on the premise, a young girl is jumping on an enclosed trampoline. Thus the premise entails the hypothesis.'
                df_train['explanation_1'][2] = 'The hypothesis states that a young human has a costume on with water coming out of his eyes, which is entailed by the premise which says a young child is dressed like a wizard crying.'
                df_train['explanation_1'][3] = 'The hypothesis says women with long brown hair work on art, which is entailed by the premise that says a couple of women with long brown hair work on an art project.'
                df_train['explanation_1'][4] = 'The hypothesis says there are three women stand around a table, the premise entails the hypothesis by stating that three women stand around a table.'
                df_train['explanation_1'][5] = 'The hypothesis says a diver is in the water, which is entailed by the premise -- a scuba diver is a type of diver, and underwater implies in the water.'
                df_train['explanation_1'][6] = 'The hypothesis making art with a pencil is a rephrase of the premise working on his art project with a pencil; thus the premise entails the hypothesis.'
                df_train['explanation_1'][7] = 'The hypothesis says there are three females, the premise states that there is one little girl and two women, one plus two is three, together there are three females; therefore the premise entails the hypothesis.'

                ## contradict
                df_train['explanation_1'][8] = 'The hypothesis says the man is using a Samsung phone, while the premise says the man is using an iPhone, an iPhone is not a Samsung phone, this is a contradiction.'
                df_train['explanation_1'][9] = 'The hypothesis says the young boy and man is sleeping at home, which conflict with the preimse that they are standing in front of a bench.'
                df_train['explanation_1'][10] = 'The hypothesis says nobody is working, but according to the premise, there are three men working, this is a contradiction.'
                df_train['explanation_1'][11] = 'The hypothesis says a women stands near a sidewalk and ready to jump rope, which contradicts with the premise that says the women stands on a rock ready to jump into the water.'
                df_train['explanation_1'][12] = 'The hypothesis says dirt bike racers are reparing their bikes, but the premise says they are racing and going fast, they cannot be repairing bikes whiling going fast, this is a contradiction.'
                df_train['explanation_1'][13] = 'The hypothesis says the person is cleaning the buliding using a red bucket, but the premise says he is using a green bucket, this is a contradiction.'
                df_train['explanation_1'][14] = 'The hypothesis says that there is no children in the park, while the premise says that children are climbing on ropes at the park, this is a contradiction.'
                df_train['explanation_1'][15] = 'The hypothesis says that people are waiting for a bus, but the premise says the people are waiting for a trolley, this is a contradiction.'

            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
    
        if args.sample_selection == 'cluster':
            if args.criteria=='hypothesis':
                df_train = pd.read_pickle('LeidenBetweenness.pkl')
            if args.criteria=='premise':
                df_train = pd.read_pickle('PremiseLeidenBetweenness.pkl')
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)

        if args.sample_selection == 'retrieval':
            if args.dataset_name == 'fm2-retrieved':
                df_train = pd.read_pickle('ReEvi_PremiseRetrieval.pkl')
                pa_tab= pa.Table.from_pandas(df_train)
                dataset['train'] = datasets.Dataset(pa_tab)
            if args.dataset_name == 'fm2':
                df_train = pd.read_pickle('fm2_HypothesisRetrieval.pkl')
                pa_tab= pa.Table.from_pandas(df_train)
                dataset['train'] = datasets.Dataset(pa_tab)
        
        if args.sample_selection == 'diversity':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/su_diversity.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)
        
        if args.sample_selection == 'fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/esnli_fastvotek_filter_'+str(args.n_shots)+'.json', lines=True)            
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)    
            
        if args.sample_selection == 'nli-fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/esnli_fastvotek_nli.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)  
            
        if args.sample_selection == 'acceptability':
            df_train = pd.read_json('/few-shot/prompt-engineering/few_shot_explanations/esnli_accept_filter_'+str(args.n_shots)+'.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab) 
        
        if args.sample_selection == 'accept-fastvotek':
            df_train = pd.read_json('/few-shot/icl-selective-annotation/esnli_accept_fastvotek_'+str(args.n_shots)+'.json', lines=True)
            pa_tab= pa.Table.from_pandas(df_train)
            dataset['train'] = datasets.Dataset(pa_tab)  
            
    if args.source_dataset_name == 'fm2':
        df_train = pd.read_csv('HypothesisLeidenDegree.csv', sep=';')
        df_train.fillna('', inplace=True)
        df_train = df_train.loc[df_train['explanation_1'] != '']
        df_train['label'] = df_train.apply(relabel, axis=1) 
        pa_tab= pa.Table.from_pandas(df_train)
        dataset['train'] = datasets.Dataset(pa_tab)
        
        
    if args.sample_selection == 'random' or args.sample_selection == 'retrieval':  ## this shuffle is for sample selection

        dataset['train'] = dataset['train'].shuffle(seed=args.sample_seed)
        dataset['validation'] = dataset['validation'].shuffle(seed=args.sample_seed)
    
        for split in ["train", "validation"]:
            split_data = dataset[split]
            label_subsets = []
            sample_size = args.n_shots if split == "train" else int(args.fewshot_eval_size/args.num_classes)
            print('sample_size: ', sample_size)
            for label in [0,1,2]:
#                 idx = [i for i, x in enumerate(split_data['label']) if x == label]
                train_examples = [sample for sample in split_data if sample['label'] == label ]
                label_subsets = label_subsets + train_examples[:sample_size]
#                 print('lable len:', len(idx))
#                 label_subset = split_data.select(idx[:sample_size]) #select `sample_size` random instances labeled as `label`
#                 label_subsets.append(label_subset) 
            tmp_df = pd.DataFrame(label_subsets)
            pa_tab= pa.Table.from_pandas(tmp_df)
            dataset[split] = datasets.Dataset(pa_tab)
#             dataset[split] = datasets.concatenate_datasets(label_subsets) #merge all label-specific instances
    
    ## this shuffle is for training robostness
    print('length of train 1:', len(dataset['train']))
    print(dataset['train'])
    dataset['train'] = dataset['train'].shuffle(seed=args.seed)
    print(dataset['train'])
    print('length of train 2:', len(dataset['train']))
    
    dataset['validation'] = dataset['validation'].shuffle(seed=args.seed) 


    original_data_splits["train"] = deepcopy(dataset["train"])
    original_data_splits["validation"] = deepcopy(dataset["validation"])
    print('length of copy:', len(original_data_splits["train"]))

    
    for split in ["train", "validation"]:
        if dataset[split] is not None:
            dataset[split] = dataset[split].map(
                lambda x: format_instance(
                    x,
                    tokenizer,
                    args.explanation_sep,
                    datasource=task_name,
                    io_format=args.io_format
                ),
                batched=False,
                load_from_cache_file=False,
            )

    data_splits["train"] = deepcopy(dataset["train"])
    print('length of copy train:', len(original_data_splits["train"]))

    data_splits["validation"] = deepcopy(dataset["validation"])

    #################
    #   TRAIN NOW   #
    #################
    training_args = TrainingArguments(
        output_dir = os.path.join(args.save_model_path, run_name), #, datetime.now().strftime("%m%d%y_%H%M%S")
        do_train=True,
        do_eval=False,
        eval_steps=args.eval_steps,
        logging_steps=1,
        save_steps = 300, # Evaluation and Save happens every 50 steps
        evaluation_strategy ='steps',
        logging_first_step=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        num_train_epochs=2,# will be ignored
        push_to_hub=False,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        lr_scheduler_type = 'constant',
        max_steps = args.max_steps,
        warmup_steps = 0,
        seed = args.seed,
        load_best_model_at_end=True,
        save_total_limit=1,
        gradient_accumulation_steps=1,
    )
        
    callbacks = [WandbCallback()]
#     callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_splits['train'],
        eval_dataset=data_splits['validation'],
        data_collator=SequenceCollator(
            model=args.model_class, pad_token=tokenizer.pad_token_id
        ),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    model = trainer.model

#     results = {}
#     train_output = trainer.evaluate(data_splits['train'])
#     perplexity = math.exp(train_output["eval_loss"])
#     results["perplexity_train"] = perplexity

#     eval_output = trainer.evaluate(data_splits['validation'])
#     perplexity = math.exp(eval_output["eval_loss"])
#     results["perplexity_validation"] = perplexity

#     print(results)

    save_path = trainer.state.best_model_checkpoint
    print('model saved at: ', save_path)
    
#     print('length of copy:', len(original_data_splits["train"]))
    original_data_splits['train'].to_csv(save_path + '/train_selet.csv')
    original_data_splits['train'].to_csv(wandb.run.dir + '/train_selet.csv')
    #################
    #   INFERENCE   #
    #################

    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.dataset_name == 'fm2':
        df_fm2 = pd.read_json('/few-shot/qafact/data/fm2/processed_test.json', lines=True)
        df_fm2 = df_fm2.rename(columns={"gold_evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['retrieved_evidence', 'id'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        df_fm2['explanation_2'] = [""]*len(df_fm2)
        df_fm2['explanation_3'] = [""]*len(df_fm2)

    if args.dataset_name == 'fm2-retrieved':
        df_fm2 = pd.read_json('/few-shot/qafact/data/fm2/processed_test.json', lines=True)
        df_fm2 = df_fm2.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['gold_evidence', 'id'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        df_fm2['explanation_2'] = [""]*len(df_fm2)
        df_fm2['explanation_3'] = [""]*len(df_fm2)
        
    if args.dataset_name == 'fever-eval':
        df_fm2 = pd.read_json('/few-shot/LOREN/data/fever/baked_data/bert_eval.json', lines=True)
        df_fm2['hypothesis'] = df_fm2['claim']
        df_fm2['premise'] = [' '.join([x[2] for x in y]) for y in df_fm2['evidence']]
        df_fm2 = df_fm2[['premise', 'hypothesis', 'label']]
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        df_fm2['explanation_2'] = [""]*len(df_fm2)
        df_fm2['explanation_3'] = [""]*len(df_fm2)
        df_fm2 = df_fm2.sample(frac=1, random_state=args.seed)[:1000]

#         df_fm2 = df_fm2[:100]

    if args.dataset_name == 'fever-train':
        df_fm2 = pd.read_json('/few-shot/LOREN/data/fever/baked_data/bert_train.json', lines=True)
        df_fm2['hypothesis'] = df_fm2['claim']
        df_fm2['premise'] = [' '.join([x[2] for x in y]) for y in df_fm2['evidence']]
        df_fm2 = df_fm2[['premise', 'hypothesis', 'label']]
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        df_fm2['explanation_2'] = [""]*len(df_fm2)
        df_fm2['explanation_3'] = [""]*len(df_fm2)
        
    if args.dataset_name == 'e-fever':
        fever_dev = pd.read_json('/few-shot/LOREN/data/fever/baked_data/bert_eval.json', lines=True)
        pub_dev = pd.read_json('/few-shot/prompt-engineering/efever/efever_dev_set.jsonl', lines=True)
        df_fm2 = pd.merge(fever_dev, pub_dev, on='id', how='inner')
        df_fm2 = df_fm2.drop(columns=['id', 'evidence'])
        df_fm2 = df_fm2.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis", 'summary': 'explanation_1'})
        ## next rows are to filter samples with explanations repeat hypothesis while the label is not entailment
        tmp = df_fm2.loc[df_fm2['label']!=0]
        tmp = tmp.loc[tmp['explanation_1']==tmp['hypothesis']]
        df_fm2 = df_fm2[ ~df_fm2.index.isin(tmp.index) ]
        ## next rows are to filter samples with wrong explanations
        tmp = df_fm2.loc[df_fm2['label']!=1]
        tmp = tmp.loc[tmp['explanation_1']=="\"The relevant information about the claim is lacking in the context.\""]
        df_fm2 = df_fm2[ ~df_fm2.index.isin(tmp.index) ]


    if args.dataset_name == 'vitaminc':
        df_fm2 = pd.read_json('/few-shot/qafact/data/vitaminc/processed_test.json', lines=True)
        # print(df_fm2)
        df_fm2['hypothesis'] = df_fm2['claim']
        df_fm2['premise'] = df_fm2['gold_evidence']
        # df_fm2['label'] = df_fm2.apply(relabel, axis=1)
        df_fm2 = df_fm2[['premise', 'hypothesis', 'label']]
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        # df_fm2 = df_fm2.drop(columns=['id'])
        df_fm2 = df_fm2.sample(frac=1, random_state=args.seed)[:1000]

        
    if args.dataset_name == 'fever-symmetric':
        df_fm2 = pd.read_json('/few-shot/prompt-engineering/fever_symmetric/test.jsonl', lines=True)
        df_fm2 = df_fm2.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['id', 'sentence1', 'sentence2', 'gold_label', 'evid'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)
#         df_fm2 = df_fm2[:20]

    if args.dataset_name == 'fever-adversarial':
        df_fm2 = pd.read_json('/few-shot/prompt-engineering/fever_adversarial/test.jsonl', lines=True)
        df_fm2 = df_fm2.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['id', 'page'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)

    if args.dataset_name == 'fever-triggers':
        df_fm2 = pd.read_json('/few-shot/prompt-engineering/fever_triggers/test.jsonl', lines=True)
        df_fm2 = df_fm2.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['id'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)

    if args.dataset_name == 'fever-gold':
        df_fm2 = pd.read_json('/few-shot/prompt-engineering/fever/test.jsonl', lines=True)
        df_fm2 = df_fm2.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        df_fm2 = df_fm2.drop(columns=['id', 'page'])
        df_fm2['explanation_1'] = [""]*len(df_fm2)
        
    df_fm2['label'] = df_fm2.apply(relabel, axis=1)

    pa_fm2 = pa.Table.from_pandas(df_fm2)
    df_test = datasets.Dataset(pa_fm2)

    input_string, answer_string = zip(*list(map(lambda x: esnli_formatting(x, args.io_format, args.explanation_sep), df_test)))
    
#     input_string = ' '.join(input_string.split())
#     answer_string = ' '.join(answer_string.split())
    
    df_test = df_test.add_column("input_string", input_string) 
    df_fm2['input_string'] = input_string
    
    df_test = df_test.add_column("answer_string", answer_string) 
    df_fm2['answer_string'] = answer_string
    
    data_set = MyDataset(df_fm2,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=args.test_bsz, shuffle=False, num_workers = 4)
    
    answers_gold = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_mask.to(device)
        answers_gold.append(model.generate(input_ids = input_ids, 
                                           attention_mask=attention_masks, 
                                           max_length=200,
                                           pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id).cpu().tolist())
        torch.cuda.empty_cache()

    decode_ans = []
    skip_special_tokens = False if "infilling" in args.io_format else True
    for i in range(len(answers_gold)):
        res = answers_gold[i]
        res = tokenizer.batch_decode(res, skip_special_tokens=skip_special_tokens)
        decode_ans = decode_ans + res

    generations_list = []
    for words in decode_ans:
        if "infilling" in args.io_format:
            words = words.replace("<extra_id_1>", f" {args.explanation_sep}")
            words = words.replace(tokenizer.pad_token,'')
            words = words.replace("<extra_id_0>", '')
            words = words.replace("<extra_id_2>", '')
            words = ' '.join(words.split())
        words = (words.replace("\n", " ").replace(tokenizer.eos_token, " ").strip())
        generations_list.append(words)

    with open(save_path + '/test_generation_batch.txt', 'w') as f:
        for line in generations_list:
            f.write(f"{line}\n")

    results = evaluate(
                        trainer.state.best_model_checkpoint,
                        df_test,
                        model,
                        tokenizer,
                        "test",
                        task_name,
                        explanation_sep=args.explanation_sep,
                        label=True,
                        rationale=args.rationale_eval,
                        device=0,
                        generations_file=save_path+'/test_generation_batch.txt',
                        io_format=args.io_format
                        )

    exp_score = evaluate_score(save_path, df_fm2, esnli_label_mapping)
    results['accept_score'] = exp_score
    
    print(results)
    wandb.log(results)
#     wandb.log(report)

#     orac = [wt5_esnli_label_mapping[x] for x in df_fm2['label']]
#     pred = [x.split(args.explanation_sep)[0] for x in generations_list]
#     target_names = ['entailment', 'neutrel', 'contradiction']
    
#     wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(preds=pred, y_true=orac)})
#     wandb.log({"classification_report" : classification_report(y_pred=pred, y_true=orac, digits=4)}) #target_names = target_names, 

              
if __name__ == "__main__":
    main()