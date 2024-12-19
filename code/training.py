import os
import wandb
os.environ["WANDB_WATCH"]='false'
os.environ["WANDB_LOG_MODEL"]='false'
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "f943c9ab18325b5cd45241ad2df308268a65c7b8"
os.environ["WANDB_CONFIG_DIR"] = "/ukp-storage-1/jyang/wandb/.config"
os.environ["WANDB_CACHE_DIR"] = "/ukp-storage-1/jyang/wandb/.cache"
os.environ["WANDB_JOB_TYPE"] = 'training'

import argparse
import json
from typing import List, Dict, Any, NewType
import numpy as np
import random 
from random import shuffle
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from evaluate import load
# from transformers import (
#     T5Config,
#     T5ForConditionalGeneration,
#     T5Tokenizer,
#     TrainingArguments,
#     set_seed
# )
from transformers.integrations import WandbCallback
import transformers

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# from transformers import Trainer
# from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from feature_conversion_methods import format_instance
from sklearn.metrics import classification_report
from utils import sep_label_explanation
import torch
import datasets
from datasets import concatenate_datasets
from load_target_dataset import load_raw_data
from copy import deepcopy 

InputDataClass = NewType("InputDataClass", Any)

# CONFIG_MAPPING = {"t5": T5Config}
# TOKENIZER_MAPPING = {"t5": T5Tokenizer}

## learning rate found through hyperparameter tuning: runing model for 50 epochs, select the best model based on acc+bertscore
## 5000 is a symbolic number that means full-set training
lr_mapping = {
    "esnli": {1: 3e-5, 2: 3e-5, 4: 3e-5, 8: 3e-5, 16: 3e-5, 32: 3e-5, 64: 3e-5, 128: 3e-5, 256: 3e-6, 512: 3e-6, 5000: 3e-6},
    "efever":{1: 3e-5, 2: 3e-5, 4: 3e-5, 8: 3e-5, 16: 3e-5, 32: 3e-5, 64: 3e-5, 128: 3e-5, 256: 3e-6, 512: 3e-6, 5000: 3e-6}
}

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
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

def format_data(dataset, task, tokenizer, explanation_sep, io_format):
    
    if dataset is not None:
        dataset = dataset.map(
            lambda x: format_instance(
                x,
                task,
                tokenizer,
                explanation_sep,
                io_format=io_format
            ),
            batched=False,
            load_from_cache_file=False,
        )

    return dataset

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def train(save_model_path, model, tokenizer, train_bsz, random_seed, data, lr, num_epochs, model_class='t5'):
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = save_model_path,
        do_train=True,
        do_eval=False,
        logging_strategy = 'no',
        save_strategy = 'no',
        evaluation_strategy ='no',
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        num_train_epochs=num_epochs,
        push_to_hub=False,
        lr_scheduler_type = 'constant',
        warmup_steps = 0,
        seed = random_seed,
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
    )
        
    callbacks = [WandbCallback()]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        data_collator=SequenceCollator(
            model=model_class, pad_token=tokenizer.pad_token_id
        ),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    trainer.train()
    
    trainer.save_model(save_model_path)

    print('Model saved at: ', save_model_path)



def train_eval(save_model_path, task, model, tokenizer, train_bsz, eval_bsz, random_seed, data, lr, num_epochs, experiment_id, model_class='t5'):    
    
    explanation_sep = ' "explanation: " '
    rouge = load('rouge', experiment_id=experiment_id)
    bertscore = load("bertscore", experiment_id=experiment_id)

    def compute_metrics(eval_preds):

        preds, labels = eval_preds # preds are predicted tokens, labels are ground truth tokens
#         print(preds)
        print(preds.shape)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
#         decoded_labels = sum(decoded_labels, [])  
        pred_l, pred_e = sep_label_explanation(decoded_preds, explanation_sep)
        gold_l, gold_e = sep_label_explanation(sum(decoded_labels,[]), explanation_sep)

        # bleu_result = bleu.compute(predictions=pred_e, references=[gold_e])
        rouge_result = rouge.compute(predictions=pred_e, references=gold_e)
        bertscore_result = bertscore.compute(predictions=pred_e, references=gold_e, model_type="microsoft/deberta-xlarge-mnli")        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        accuracy = sum([pred_l[i] == gold_l[i] for i in range(len(pred_l))])/len(pred_l)

        # result = {'bleu' : bleu_result['score']}
        result= {"gen_len": np.mean(prediction_lens)}
        result["rouge1"] = np.mean(rouge_result["rouge1"])
        result["rouge2"] = np.mean(rouge_result["rouge2"])
        result["rougeL"] = np.mean(rouge_result["rougeL"])
        result["rougeLsum"] = np.mean(rouge_result["rougeLsum"])
        result["bertscore"] = np.mean(bertscore_result["f1"])
        result["accuracy"] = accuracy
        result['acc_bertscore'] = accuracy + result["bertscore"]
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = save_model_path,
        do_train=True,
        do_eval=True,
        logging_strategy = 'epoch',
        evaluation_strategy ='epoch',
        save_strategy = 'epoch',
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=eval_bsz,
        num_train_epochs=num_epochs,
        push_to_hub=False,
        metric_for_best_model = 'eval_eval_acc_bertscore',
        greater_is_better = True,
        lr_scheduler_type = 'constant',
        warmup_steps = 0,
        seed = random_seed,
        save_total_limit = 1,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
        predict_with_generate=True
    )
        
    callbacks = [WandbCallback()]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset={'train': data['train'], 'eval':data['validation']},
        data_collator=SequenceCollator(
            model=model_class, pad_token=tokenizer.pad_token_id
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    trainer.train()
    
    trainer.save_model(save_model_path)
    
    print('Model saved at: ', save_model_path)

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='few-shot-train-recod', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default= 't5-large', type=str, required=False, 
                       help="model and tokenizer name")
    parser.add_argument("--random_seed", default= 42, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, 
                       help="subset dataset")
    parser.add_argument("--sample_seed", default= 0, type=int, required=False, 
                       help="sample selection seed")
    parser.add_argument("--lr", default= 3e-5, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--train_bsz", default= 4, type=int, required=False, 
                       help="training batch size")
    parser.add_argument("--eval_bsz", default= 16, type=int, required=False, 
                       help="training batch size")                   
    parser.add_argument("--n_shots", default= 8, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--explanation_source", default= 'na', type=str, required=False, 
                       help="source of explanation")                   
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--io_format", default= 'standard', type=str, required=False, 
                       help="nli prompt format")
    parser.add_argument("--save_model_path", default= '../model/', type=str, required=False, 
                       help="path to save model")
#     parser.add_argument("--max_steps", default= 10000, type=int, required=False, 
#                        help="number of training steps")
#     parser.add_argument("--eval_steps", default= 200, type=int, required=False, 
#                        help="number of training steps")
    parser.add_argument("--num_epochs", default= 50, type=int, required=False, 
                       help="number of training epochs")
    parser.add_argument("--do_eval", action='store_true',
                       help="whether perform validation or not")
    parser.add_argument("--tuning", action='store_true',
                       help="tuning model")
    
    args = parser.parse_args()
    
    if args.tuning:
        learning_rate = args.lr
    else:    
        learning_rate = lr_mapping[args.source_dataset_name][args.n_shots]

    run_name = "/".join((args.source_dataset_name, args.sample_selection, 'sub'+str(args.data_sub), 'nt'+str(args.n_shots)))
    
    if args.tuning:
        run_name = "+".join((run_name, 'lr'+str(args.lr)))
        
    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.sample_selection,'train', args.source_dataset_name],
           config = args,
           save_code = True)    

    set_seed(args.random_seed)
    set_other_seeds(args.random_seed)

    print('Data loading...')    
    data_path =  "/".join(('../samples',args.source_dataset_name, args.sample_selection, 'sub_'+str(args.data_sub), str(args.n_shots)))
    
    if args.n_shots == 5000:
        data_train = load_raw_data(args.source_dataset_name, split ='train')
    elif args.sample_selection == 'specific':    
        if args.explanation_source == 'human':
            data_train = load_raw_data(args.source_dataset_name, split="human_exp")  
        if args.explanation_source == 'original':   
            data_train = load_raw_data(args.source_dataset_name, split="orig_exp")  
    elif args.sample_selection == 'mixed_source': 
        tmp_path1 =  "/".join(('../samples','esnli', 'accept-fastvotek', 'sub_'+str(args.data_sub), str(args.n_shots//2)))
        train_1 = datasets.load_dataset("json", data_files=tmp_path1+"/train_select.json", split="train") 
        tmp_path2 =  "/".join(('../samples','efever', 'accept-fastvotek', 'sub_'+str(args.data_sub), str(args.n_shots//2)))
        train_2 = datasets.load_dataset("json", data_files=tmp_path2+"/train_select.json", split="train") 
        data_train = concatenate_datasets([train_1, train_2])
        data_train = data_train.remove_columns(['explanation_2','explanation_3'])
    else:    
        data_train = datasets.load_dataset("json", data_files=data_path+"/train_select.json", split="train")  

    if args.do_eval:
        if args.sample_selection == 'mixed_source': 
            data_eval_esnli = datasets.load_dataset("json", data_files="../datasets/source/esnli/val_select.json", split="train")  
            data_eval_efever = datasets.load_dataset("json", data_files="../datasets/source/efever/val_select.json", split="train")  
            data_eval = concatenate_datasets([data_eval_esnli, data_eval_efever])
            data_eval = data_eval.remove_columns(['explanation_2','explanation_3'])
        else:    
            data_eval = datasets.load_dataset("json", data_files="../datasets/source/"+args.source_dataset_name+"/val_select.json", split="train")  

    print('Model initializing...')

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
    
    print('Data formatting...')
    data_splits = {'train': None, 'validation': None, 'test': None}

    data_splits['train'] = deepcopy(format_data(
        dataset=data_train, 
        task=args.source_dataset_name,
        tokenizer = tokenizer, 
        explanation_sep=args.explanation_sep, 
        io_format=args.io_format
    ))

    if args.do_eval:
        data_splits['validation'] = deepcopy(format_data(
            dataset=data_eval, 
            task=args.source_dataset_name,
            tokenizer=tokenizer, 
            explanation_sep=args.explanation_sep, 
            io_format=args.io_format
        ))

    print('Start training...')
    save_model_path = '/'.join((args.save_model_path,run_name))

    if not args.do_eval:
        train(
            save_model_path, 
            model=model, 
            tokenizer = tokenizer, 
            train_bsz=args.train_bsz, 
            random_seed=args.random_seed, 
            data=data_splits, 
            lr=learning_rate, 
            num_epochs=args.num_epochs
        )
    else:    
        train_eval(
            save_model_path, 
            task=args.source_dataset_name,
            model=model, 
            tokenizer = tokenizer, 
            train_bsz=args.train_bsz, 
            eval_bsz=args.eval_bsz,
            random_seed=args.random_seed, 
            data=data_splits, 
            lr=learning_rate, 
            experiment_id=args.n_shots,
            num_epochs=args.num_epochs
        )

    print('Finished training.')
    
    wandb.finish()
                      
if __name__ == "__main__":
    main()