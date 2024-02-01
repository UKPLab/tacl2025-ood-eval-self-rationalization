import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    set_seed
)
import datasets
from load_target_dataset import load_format_data
# from transformers import BartTokenizer, BartForConditionalGeneration

import torch
from feature_conversion_methods import label_mapping
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

def formatting(item, dataset_name, io_format='standard', explanation_sep=' "explanation: " '):

    premise = item["premise"]
    hypothesis = item["hypothesis"]
    answer = label_mapping[dataset_name][item["label"]]
    abstr_expl = item["explanation_1"].lower() 
    # Dev/test instances have more than one explanation annotated; merge them into one sequence separated by [SEP] 
    for k in [2,3]:
        if f"explanation_{k}" in item and item[f'explanation_{k}']!='': 
            abstr_expl += f" [SEP] {item[f'explanation_{k}'].lower()}"

    if io_format == 'standard':
        input_string = f"explain nli hypothesis: {hypothesis} premise: {premise}"
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"
    
    if io_format == 'flan':
        #
        input_string = f"Given hypothesis: {hypothesis} and premise: {premise}, predict if the premise entails or contradicts or is neutral to the hypothesis."
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"
        
    return input_string, answer_string

class MyDataset(Dataset):
    def __init__(self, df, tokenizer):

        input_data = tokenizer.batch_encode_plus(df['input_string'].to_list(), 
#                                            max_length = 200, 
                                           return_tensors="pt", 
                                           padding=True, 
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                          )
        
#         dec = tokenizer.batch_encode_plus(
#                                     df['answer_string'].to_list(),
# #                                     max_length=200,
#                                     return_tensors="pt", 
#                                     padding=True,
#                                     return_token_type_ids=False,
#                                     return_attention_mask=True,
#                                 )
        
        self.input_ids = input_data['input_ids']
        self.attention_mask =input_data['attention_mask']
        
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
    
    def __len__(self):
        return len(self.input_ids)

model_name = 't5-large'

tokenizer = T5Tokenizer.from_pretrained(model_name,local_files_only=False)
model = T5ForConditionalGeneration.from_pretrained(model_name,local_files_only=False)

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

seen_list = ['sick','add_one','joci','mpe','dnc','hans','wnli','glue_diagnostics','conj',
             'vitaminc','covid_fact','fm2','snopes_stance','scifact','climate-fever-combined',
             'xsum_hallucination', 'factcc', 'qags_cnndm', 'qags_xsum'] 

data_esnli = load_format_data('esnli', 'val')
data_efever = load_format_data('efever', 'val')

data_list = {}
for name in seen_list:
    data_list[name] = load_format_data(name, 'test')

def shuffle_data(data, size=360):
    data = data.shuffle(seed=42)
    if len(data)>=size:
        data = data.select(range(0, size))
    return data

def get_embeddings(dataset, batch_size):
    df_data = dataset.to_pandas()
    data_set = MyDataset(df_data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers = 1)
    pred_embs = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model.encoder(input_ids = input_ids, attention_mask=attention_mask, return_dict=True)
        emb = (output.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=-2) /attention_mask.unsqueeze(-1).sum(dim=-2)
        pred_embs +=(emb.cpu().detach().tolist())
        #     pred_text = pred_text + (tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True))
    #     pred_prob = pred_prob + (torch.exp(transition_scores)[:,0].cpu().tolist())
        torch.cuda.empty_cache()
    return pred_embs

def cosine_distance(arr1, arr2):
    distance = cosine(arr1, arr2)
    return distance

def euclidean_distance(arr1, arr2):
    squared_diff = np.square(arr1 - arr2)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance

for name in seen_list:
    data_list[name] = shuffle_data(data_list[name])

seen_large = ['snopes_stance','scifact','climate-fever-combined','covid_fact', 'xsum_hallucination', 'factcc', 'qags_cnndm', 'qags_xsum'] 

seen_small = ['sick','add_one','joci','mpe','dnc','hans','wnli','glue_diagnostics','conj','vitaminc','fm2'] 

esnli_embs = get_embeddings(data_esnli, batch_size=32)
efever_embs = get_embeddings(data_efever, batch_size=16)
embs_list = {}
for name in seen_small:
    print(name)
    embs_list[name] = get_embeddings(data_list[name],batch_size=32)

for name in seen_large:
    print(name)
    embs_list[name] = get_embeddings(data_list[name],batch_size=1)

esnli_emb_matrix = np.array(esnli_embs)
efever_emb_matrix = np.array(efever_embs)
for name in seen_small:
    # print(name)
    embs_list[name] = np.array(embs_list[name])

for name in seen_large:
    # print(name)
    embs_list[name] = np.array(embs_list[name])  

efever_mean = np.mean(efever_emb_matrix, axis=0)
esnli_mean = np.mean(esnli_emb_matrix, axis=0)
for name in seen_small:
#     print(name)
    embs_list[name] = np.mean(embs_list[name], axis=0)

for name in seen_large:
#     print(name)
    embs_list[name] = np.mean(embs_list[name], axis=0)

from scipy.spatial.distance import cosine

print('cosine similarity with esnli:')
for name in seen_small:
    print(name)
    distance = cosine_distance(embs_list[name], esnli_mean)
    print(distance)

for name in seen_large:
    print(name)
    distance = cosine_distance(embs_list[name], esnli_mean)
    print(distance)

print('cosine similarity with efever:')
for name in seen_small:
    print(name)
    distance = cosine_distance(embs_list[name], efever_mean)
    print(distance)

for name in seen_large:
    print(name)
    distance = cosine_distance(embs_list[name], efever_mean)
    print(distance)