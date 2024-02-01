import argparse
import os
from random import shuffle
import pandas as pd
import pyarrow as pa
import numpy as np
import datasets
from tqdm import tqdm
import torch
from utils import format_example, calculate_sentence_transformer_embedding, fast_votek
from Inference import MyDataset
from feature_conversion_methods import formatting
from torch.utils.data import Dataset, DataLoader
from nli_demo import get_scores
import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
)

label2text = {0:'entailment', 1:'neutral', 2:'contradiction'}
efever_label_mapping = {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2}

def random_select(data, seed, sample_size):
    data = data.shuffle(seed=seed)
    labels = data.features['label'].names
    label_subsets = []
    for label in labels:
        label_int = data.features['label'].str2int(label)
        train_examples = [sample for sample in data if sample['label'] == label_int ]
        label_subsets = label_subsets + train_examples[:sample_size]
    tmp_df = pd.DataFrame(label_subsets)
    pa_tab= pa.Table.from_pandas(tmp_df)
    data = datasets.Dataset(pa_tab)
    # data.to_csv(save_path + '/train_selet.csv')
    return data

def fastvotek(data, save_dir):

    embedding_model = 'sentence-transformers/paraphrase-mpnet-base-v2'

    data_sample = [e for e in data]
    
    sample_0 = [sample for sample in data_sample if sample['label'] == 0 ]
    sample_1 = [sample for sample in data_sample if sample['label'] == 1 ]
    sample_2 = [sample for sample in data_sample if sample['label'] == 2 ]

    sample_text_0 = [format_example(row) for row in sample_0]
    sample_text_1 = [format_example(row) for row in sample_1]
    sample_text_2 = [format_example(row) for row in sample_2]

    embeddings_0 = calculate_sentence_transformer_embedding(text_to_encode=sample_text_0, embedding_model = embedding_model)
    embeddings_1 = calculate_sentence_transformer_embedding(text_to_encode=sample_text_1, embedding_model = embedding_model)
    embeddings_2 = calculate_sentence_transformer_embedding(text_to_encode=sample_text_2, embedding_model = embedding_model)
    
    tmp_dir = save_dir

    # we save all data at the same time so we don't repeatively calculate the embeddings
    for n_shot in tqdm([1, 2]): #1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        selected_indices_0 = fast_votek(embeddings_0, batch_size=n_shot)
        selected_indices_1 = fast_votek(embeddings_1, batch_size=n_shot)
        selected_indices_2 = fast_votek(embeddings_2, batch_size=n_shot)

        sample_select_0 = [sample_0[ind] for ind in selected_indices_0]
        sample_select_1 = [sample_1[ind] for ind in selected_indices_1]
        sample_select_2 = [sample_2[ind] for ind in selected_indices_2]

        sample_select = sample_select_0 + sample_select_1 + sample_select_2

        df=pd.DataFrame(sample_select)
        save_dir = tmp_dir + '/' + str(n_shot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_json(save_dir + '/train_select.json', orient='records', lines=True)
        print("succefully saved!")

def acceptability(data, save_dir, thres = 0.5): #this function is to filter unacceptable explanations
    df_data = data.to_pandas()
    inputs = ['premise: '+ row['premise'] + ' hypothesis: ' + row['hypothesis'] + ' answer: '+ label2text[row['label']] + ' explanation: ' + row['explanation_1'] for _, row in df_data.iterrows()]   
    model_type = 'large'
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scores = get_scores(
            inputs,
            model_type,
            batch_size=batch_size,
            device=device,
            verbose=False
    )
    df_data['exp_score'] = scores
    df_data = df_data.loc[df_data['exp_score']>thres]
    df_data = df_data.sort_values(by=['exp_score'], ascending=False)
    del df_data['exp_score']

    pa_tab= pa.Table.from_pandas(df_data)
    data = datasets.Dataset(pa_tab)

    return data
    # fastvotek(data, save_dir)

def least_confidence(data, model_path, save_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_path,local_files_only=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path,local_files_only=False)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    input_string, answer_string = zip(*list(map(lambda x: formatting(x, 'esnli'), data)))
    data = data.add_column("input_string", input_string)
    data = data.add_column("answer_string", answer_string) 
    df_data = data.to_pandas()
    data_set = MyDataset(df_data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=False, num_workers = 2)

    pred_text = []
    pred_prob = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        output = model.generate(input_ids = input_ids, output_scores=True, return_dict_in_generate=True)
        transition_scores = model.compute_transition_scores(sequences= output.sequences, scores=output.scores, normalize_logits=True)
        pred_text = pred_text + (tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True))
        pred_prob = pred_prob + (torch.exp(transition_scores)[:,0].cpu().tolist())
        torch.cuda.empty_cache()

    df_data["pred_text"] = pred_text   
    df_data["pred_prob"] = pred_prob   

    tmp_dir = save_dir

    sorted_df = df_data.sort_values('pred_prob')
    for n_shot in tqdm([1, 2, 4, 8, 16, 32, 64, 128]): #4, 8, 16, 32, 64, 128, 256, 512
        sorted_df0 = sorted_df.loc[sorted_df['label']==0]
        sorted_df0=sorted_df0.iloc[:n_shot]

        sorted_df1 = sorted_df.loc[sorted_df['label']==1]
        sorted_df1=sorted_df1.iloc[:n_shot]

        sorted_df2 = sorted_df.loc[sorted_df['label']==2]
        sorted_df2=sorted_df2.iloc[:n_shot]

        sorted_dfc = pd.concat([sorted_df0, sorted_df1, sorted_df2])
    
        save_dir = tmp_dir + '/' + str(n_shot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sorted_dfc.to_json(save_dir + '/train_select.json', orient='records', lines=True)
        print("succefully saved!")

def most_confidence(data, model_path, save_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_path,local_files_only=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path,local_files_only=False)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    input_string, answer_string = zip(*list(map(lambda x: formatting(x, 'esnli'), data)))
    data = data.add_column("input_string", input_string)
    data = data.add_column("answer_string", answer_string) 
    df_data = data.to_pandas()
    data_set = MyDataset(df_data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=False, num_workers = 2)

    pred_text = []
    pred_prob = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        output = model.generate(input_ids = input_ids, output_scores=True, return_dict_in_generate=True)
        transition_scores = model.compute_transition_scores(sequences= output.sequences, scores=output.scores, normalize_logits=True)
        pred_text = pred_text + (tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True))
        pred_prob = pred_prob + (torch.exp(transition_scores)[:,0].cpu().tolist())
        torch.cuda.empty_cache()

    df_data["pred_text"] = pred_text   
    df_data["pred_prob"] = pred_prob   

    tmp_dir = save_dir

    sorted_df = df_data.sort_values('pred_prob', ascending=False)
    for n_shot in tqdm([1, 2]): #4, 8, 16, 32, 64, 128, 256, 512
        sorted_df0 = sorted_df.loc[sorted_df['label']==0]
        sorted_df0=sorted_df0.iloc[:n_shot]

        sorted_df1 = sorted_df.loc[sorted_df['label']==1]
        sorted_df1=sorted_df1.iloc[:n_shot]

        sorted_df2 = sorted_df.loc[sorted_df['label']==2]
        sorted_df2=sorted_df2.iloc[:n_shot]

        sorted_dfc = pd.concat([sorted_df0, sorted_df1, sorted_df2])
    
        save_dir = tmp_dir + '/' + str(n_shot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sorted_dfc.to_json(save_dir + '/train_select.json', orient='records', lines=True)
        print("succefully saved!")

def ambiguous(data, model_path, save_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_path,local_files_only=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path,local_files_only=False)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    input_string, answer_string = zip(*list(map(lambda x: formatting(x, 'esnli'), data)))
    data = data.add_column("input_string", input_string)
    data = data.add_column("answer_string", answer_string) 
    df_data = data.to_pandas()
    data_set = MyDataset(df_data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=False, num_workers = 2)

    pred_text = []
    pred_prob = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        output = model.generate(input_ids = input_ids, output_scores=True, return_dict_in_generate=True)
        transition_scores = model.compute_transition_scores(sequences= output.sequences, scores=output.scores, normalize_logits=True)
        pred_text = pred_text + (tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True))
        pred_prob = pred_prob + (torch.exp(transition_scores)[:,0].cpu().tolist())
        torch.cuda.empty_cache()

    df_data["pred_text"] = pred_text   
    df_data["pred_prob"] = pred_prob   

    tmp_dir = save_dir

    # sorted_df = df_data.sort_values('pred_prob',ascending=False)

    median_value = (df_data['pred_prob'].max() + df_data['pred_prob'].min())/2
    df_data['abs_median_preb'] = np.abs(df_data['pred_prob'] - median_value)
    sorted_df = df_data.sort_values('abs_median_preb',ascending=True)

    for n_shot in tqdm([1, 2, 4, 8, 16, 32, 64, 128]): #
        sorted_df0 = sorted_df.loc[sorted_df['label']==0]
        sorted_df0=sorted_df0.iloc[:n_shot]

        sorted_df1 = sorted_df.loc[sorted_df['label']==1]
        sorted_df1=sorted_df1.iloc[:n_shot]

        sorted_df2 = sorted_df.loc[sorted_df['label']==2]
        sorted_df2=sorted_df2.iloc[:n_shot]

        sorted_dfc = pd.concat([sorted_df0, sorted_df1, sorted_df2])
    
        save_dir = tmp_dir + '/' + str(n_shot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sorted_dfc.to_json(save_dir + '/train_select.json', orient='records', lines=True)
        print("succefully saved!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", default='esnli', type=str, required=False, 
                       help="source dataset")
    parser.add_argument("--sample_selection", default='random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--subset_size", default=5000, type=int, required=False, 
                       help="size for the subset selection")  
    parser.add_argument("--split", default=0, type=int, required=False, 
                       help="split for the subset selection") 
    parser.add_argument("--n_shots", default=8, type=int, required=False, 
                       help="number of samples per class")  
    parser.add_argument("--model_path", default='../model', type=str, required=False, 
                       help="model path is required for least_confidence method")
    parser.add_argument("--output_dir", default='../samples/', type=str, required=False, 
                       help="output directory")                   
    args = parser.parse_args()
    
    print("Loading data...")
    if args.source_dataset == 'esnli':
    # First random select a subset, we fix the seed for this selection
        data = datasets.load_dataset(args.source_dataset)
        data_sample = data["train"].shuffle(seed=42)
#         if len(data_sample) > args.subset_size:
        data_sample = data_sample.select(range(args.split*args.subset_size, (args.split+1)*args.subset_size))

    if args.source_dataset == 'anli':
        data = datasets.load_dataset(args.source_dataset)
        data = datasets.concatenate_datasets([data["dev_r1"], data["dev_r2"], data["dev_r3"]])
        data_sample = data.shuffle(seed=42) 
        if len(data_sample) > args.subset_size:
            data_sample = data_sample.select(range(args.subset_size))
        data_sample = data_sample.rename_column("reason", "explanation_1")

    if args.source_dataset == 'efever':    
        fever = pd.read_json('../datasets/source/efever/fever_train.jsonl', lines=True)
        efever = pd.read_json('../datasets/source/efever/efever_train_set.jsonl', lines=True)
        df_data = pd.merge(fever, efever, on='id', how='inner')
        df_data = df_data.drop(columns=['id', 'verifiable', 'evidence'])
        df_data = df_data.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis", 'summary': 'explanation_1'})
        labels = [efever_label_mapping[row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels
        ## next rows are to filter samples with explanations repeat hypothesis while the label is not entailment
        tmp = df_data.loc[df_data['label']!=0]
        tmp = tmp.loc[tmp['explanation_1']==tmp['hypothesis']]
        df_data = df_data[ ~df_data.index.isin(tmp.index) ]
        ## next rows are to filter samples with wrong explanations
        tmp = df_data.loc[df_data['label']!=1]
        tmp = tmp.loc[tmp['explanation_1']=="\"The relevant information about the claim is lacking in the context.\""]
        df_data = df_data[ ~df_data.index.isin(tmp.index)]

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

        data_sample = data.shuffle(seed=42)
        data_sample = data_sample.select(range(args.split*args.subset_size, (args.split+1)*args.subset_size))

        data_sample = data_sample.remove_columns('__index_level_0__')

#     if args.sample_selection == 'random':
#         save_dir = '/'.join((args.output_dir, args.source_dataset, args.sample_selection, 'sub_'+str(args.split)))
#     else:
    save_dir = '/'.join((args.output_dir, args.source_dataset, args.sample_selection, 'sub_'+str(args.split)))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.sample_selection == 'random':
        #generate 10 random seeds for sample selection
#         seeds = [x for x in range(0, 10)]
#         for s in tqdm(seeds):
        tmp_dir = save_dir
        sample_seed = 0
        for ns in [1,2]:#,4,8,16,32,64,128,256,512
            sample = random_select(data_sample, sample_seed, ns)
            save_dir = tmp_dir + '/' + str(ns)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample.to_json(save_dir + '/train_select.json', orient='records', lines=True)
            print("succefully saved!")

    if args.sample_selection == 'fastvotek':
        fastvotek(data_sample, save_dir)

    if args.sample_selection == 'accept-fastvotek':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3)
        fastvotek(data_sample, save_dir)

    if args.sample_selection == 'ambiguous':
        ambiguous(data_sample, args.model_path, save_dir) 

    if args.sample_selection == 'accept-ambiguous':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3) 
        ambiguous(data_sample, args.model_path, save_dir) 

    if args.sample_selection == 'most_confidence':
        most_confidence(data_sample, args.model_path, save_dir) 

    if args.sample_selection == 'least_confidence':
        least_confidence(data_sample, args.model_path, save_dir)

    if args.sample_selection == 'accept-least_confidence':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3) 
        least_confidence(data_sample, args.model_path, save_dir)     

if __name__ == "__main__":
    main()

