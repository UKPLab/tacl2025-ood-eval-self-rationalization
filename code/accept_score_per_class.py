# import pandas as pd

# def accept_per_class(source, target):
#     result_path = '../result/'+target+'/'+source+'/accept-fastvotek/sub0/nt64/exp_correct_scores.json'
#     df_exp = pd.read_json(result_path, lines=True)
#     print('entailment:',df_exp.loc[df_exp['label']==' entailment']['accept_score'].mean())
#     print('neutral:',df_exp.loc[df_exp['label']==' neutral']['accept_score'].mean())
#     print('contradiction:',df_exp.loc[df_exp['label']==' contradiction']['accept_score'].mean())
#     print('not_entailment:',df_exp.loc[df_exp['label']==' not_entailment']['accept_score'].mean())

# target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc',
#                    'climate-fever-combined','snopes_stance','covid_fact',"scifact",
#                    'factcc','xsum_hallucination','qags_xsum','qags_cnndm']

# for dataset in target_datasets:
#     print("dataset:",dataset)
#     print("esnli:")
#     accept_per_class('esnli',dataset)
#     print("efever:")
#     accept_per_class('efever',dataset)

import pandas as pd
import json

def accept_per_class(source, target):
    result_path = '../result/'+target+'/'+source+'/accept-fastvotek/sub0/nt64/exp_correct_scores.json'
    df_exp = pd.read_json(result_path, lines=True)
    
    return {
        'entailment': df_exp.loc[df_exp['label']==' entailment']['accept_score'].mean(),
        'neutral': df_exp.loc[df_exp['label']==' neutral']['accept_score'].mean(),
        'contradiction': df_exp.loc[df_exp['label']==' contradiction']['accept_score'].mean(),
        'not_entailment': df_exp.loc[df_exp['label']==' not_entailment']['accept_score'].mean()
    }

target_datasets = ['wnli', 'add_one', 'glue_diagnostics', 'fm2', 'mpe', 'joci', 'hans', 'conj', 'dnc', 'sick', 'vitaminc',
                   'climate-fever-combined', 'snopes_stance', 'covid_fact', "scifact",
                   'factcc', 'xsum_hallucination', 'qags_xsum', 'qags_cnndm']

all_results = {}

for dataset in target_datasets:
    all_results[dataset] = {
        'esnli': accept_per_class('esnli', dataset),
        'efever': accept_per_class('efever', dataset)
    }

# Filter out individual NaN scores but keep datasets and sources
for dataset in all_results.keys():
    for source in all_results[dataset].keys():
        all_results[dataset][source] = {k: v for k, v in all_results[dataset][source].items() if pd.notna(v)}

# Convert the results to a JSON string
results_json = json.dumps(all_results, indent=4)

# Write the JSON string to a file
with open('../result/accept_per_class.json', 'w') as file:
    file.write(results_json)

# import pandas as pd
# import json
# from nli_demo import get_scores
# import torch
# from feature_conversion_methods import label_mapping
# model_type = 'large'
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# df_efever = pd.read_json("../samples/efever/accept-fastvotek/sub_0/64/train_select.json",lines=True)
# df_esnli = pd.read_json("../samples/esnli/accept-fastvotek/sub_0/64/train_select.json",lines=True)

# def cal_accept(df,task):
#     inputs = ['premise: '+ row['premise'] + ' hypothesis: ' + row['hypothesis'] + ' answer: '+ label_mapping[task][row['label']] + ' explanation: ' + row['explanation_1'] for _, row in df.iterrows()]   

#     scores = get_scores(
#     inputs,
#     model_type,
#     batch_size=32,
#     device=device,
#     verbose=False)
#     return scores

# df_efever['accept_score']=cal_accept(df_efever,'efever')
# df_esnli['accept_score']=cal_accept(df_esnli,'esnli')

# def accept_per_class(df):
#     entailment=df.loc[df['label']==0]['accept_score'].mean()
#     neutral=df.loc[df['label']==1]['accept_score'].mean()
#     contradiction=df.loc[df['label']==2]['accept_score'].mean()
#     return entailment, neutral, contradiction

# # Writing results to a text file
# with open('../result/source_accept_per_class.txt', 'w') as file:
#     file.write(f"source: esnli\n")
#     entailment, neutral, contradiction = accept_per_class(df_esnli)
#     file.write(f"entailment: {entailment}:\n")
#     file.write(f"neutral: {neutral}:\n")
#     file.write(f"contradiction: {contradiction}:\n")
#     file.write("\n")

#     file.write(f"source: efever\n")
#     entailment, neutral, contradiction = accept_per_class(df_efever)
#     file.write(f"entailment: {entailment}:\n")
#     file.write(f"neutral: {neutral}:\n")
#     file.write(f"contradiction: {contradiction}:\n")
#     file.write("\n")