import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    set_seed
)
from metrics import evaluate
from nli_demo import evaluate_score

from load_target_dataset import load_format_data
from Inference import inference

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 
    
CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}

def wse(model_0, model_1, alpha):
    # load state dicts from checkpoints
    theta_0 = model_0.state_dict()
    theta_1 = model_1.state_dict()
    
    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())
    # interpolate between all weights in the checkpoints
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    # update the model (in-place) according to the new weights
    #model.load_state_dict(theta)
    return theta

set_seed(42)
set_other_seeds(42)

# model_name = 't5-large'
model_path_efever = '../model/efever/accept-ambiguous/sub0/nt64/'
model_class='t5'
model_path_esnli = '../model/esnli/accept-fastvotek/sub0/nt64/'

## load best model
tokenizer_name = TOKENIZER_MAPPING[model_class]
tokenizer = tokenizer_name.from_pretrained(model_path_esnli, local_files_only=False)
model_best_esnli = T5ForConditionalGeneration.from_pretrained(model_path_esnli, local_files_only=False)

model_best_efever = T5ForConditionalGeneration.from_pretrained(model_path_efever, local_files_only=False)

new_theta = wse(model_best_esnli, model_best_efever, 0.5)

model_best_efever.load_state_dict(new_theta)

target_datasets = ['climate-fever-combined','snopes_stance','covid_fact',"scifact",'factcc','xsum_hallucination','qags_xsum','qags_cnndm']

for dataset in target_datasets:
    df_result=pd.DataFrame()
    result_path = '/'.join(('../result', dataset, 'esnli/ensamble/'))
    
    data = load_format_data(dataset, 'test')
    
    labels, explanations, label_probabilities = inference(
        model=model_best_efever, 
        tokenizer=tokenizer,
        seed=42, 
        data=data, 
        test_bsz=12, 
        result_path=result_path,
        explanation_sep=' "explanation: " '
    )
    
    results, cm = evaluate(
        result_path,
        data,
        tokenizer,
        "test",
        task=dataset,
        labels=labels,
        explanations=explanations
    )
    df_data = data.to_pandas()
    exp_score = evaluate_score(result_path, df_data, dataset, 32)
    results['accept_score'] = exp_score
    
    label_probabilities=np.asarray(label_probabilities)

    df_result=pd.DataFrame()
    df_result['label']=labels
#     df_result['explanation']=explanations
    df_result['prob_en']=label_probabilities[:,0]
    df_result['prob_neutral']=label_probabilities[:,1]
    df_result['prob_contradiction']=label_probabilities[:,2]
    df_result.to_json(result_path+'/l_e_prob.json', orient='records', lines=True)
    
    print(dataset)
    print(results)
