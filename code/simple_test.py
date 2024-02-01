# # from load_target_dataset import load_raw_data
# # import datasets
# # import re
# # URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')
# # from datasets import Dataset
# # import wandb
# # import os
# # import numpy as np
# # import pandas as pd

# # data = load_raw_data('qags_xsum')
# # data = load_raw_data('qags_cnndm')

# # # data = datasets.concatenate_datasets([data, data1])
# # df_agg = pd.read_csv('../datasets/target/hallucination/aggre_fact_final.csv')
# # df_frank =  df_agg.loc[(df_agg['dataset']=='FRANK') & (df_agg['cut']=='test')]
# # data = datasets.Dataset.from_pandas(df_frank)

# # cols_to_remove = data.column_names
# # cols_to_remove.remove("doc") 
# # cols_to_remove.remove("summary")
# # cols_to_remove.remove("label")
# # data = data.remove_columns(cols_to_remove)
# # data = data.rename_column("summary", "hypothesis")
# # data = data.rename_column("doc", "premise")
# # new_column = [""] * len(data)
# # data = data.add_column("explanation_1", new_column) 
# # # data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

# # print(df_data)
# # print(df_evidence)
# # print(data)
# # # print([len(c.split()) for c in data['hypothesis']])
# # len_claim = np.mean([len(c.split()) for c in data['hypothesis']])
# # len_evidence = np.mean([len(c.split()) for c in data['premise']])
# # print(len_claim)
# # print(len_evidence)
# # print(data['hypothesis'][110])
# # print(data['premise'][110])
# # print(data['label'][110])

# import os
# import json
# import pandas as pd

# source_datasets=['na']
# # n_shots = [1,2,4,8,16,32,64,128]
# n_shots=[0]
# # sub_set=[0]
# # sample_selection=['random']
# # sample_selection = ['ambiguous','accept-ambiguous','fastvotek','accept-fastvotek','least_confidence']#,'ambiguous','accept-ambiguous'
# # sub_set = [0,1,2,3,4]#

# target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc',
#                    'climate-fever-combined','snopes_stance','covid_fact',"scifact",
#                    'factcc','xsum_hallucination','qags_xsum','qags_cnndm']
  

# def load_result(d):
#     if os.path.exists(d +'/results_test.json'):
#         f = open(d +'/results_test.json')
#     return json.load(f)  

    
# dirs=[]
# sources=[]
# targets=[]
# selections=[]
# shots=[]
# subsets=[]
# accs=[]
# f1s=[]
# accepts=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for selection in sample_selection:
#             for ns in n_shots: 
#                 for ss in sub_set:
#                     path = '/'.join(('../result',
#                                             dataset,source,selection,'sub'+str(ss),'nt'+str(ns)))

#                     print(path)
#                     data=load_result(path)
# #                     if source=='efever':
# #                         print(path)
#                     dirs.append(path)
#                     sources.append(source)
#                     targets.append(dataset)
#                     selections.append(selection)
#                     shots.append(ns)
#                     subsets.append(ss)
#                     accs.append(data['test_acc'])
#                     f1s.append(data['test_macro_avg_f1'])
#                     accepts.append(data['accept_score_correct_only'])



# dict_prepare = {

#     'target_dataset':targets,
#     'source_dataset':sources,
#     'sample_selection':selections,
#     'n_shots':shots,
#     'sub_set':subsets,
#     'accuracy':accs,
#     'F1_score':f1s,
#     'accept_score':accepts,
# }

# df = pd.DataFrame(dict_prepare, columns = ['target_dataset','source_dataset','sample_selection', 'n_shots',
#                                                'sub_set','accuracy','F1_score','accept_score'])


# df.to_json( '../result/full-shot-efever-slurm.json', orient='records', lines=True)

# dirs=[]
# sources=[]
# targets=[]
# shots=[]
# accs=[]
# f1s=[]
# # accepts=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for ns in n_shots: 
#             path = '/'.join(('../result/zero_shot',
#                     dataset))

#             print(path)
#             data=load_result(path)
# #                     if source=='efever':
# #                         print(path)
#             dirs.append(path)
#             sources.append(source)
#             targets.append(dataset)
#             shots.append(ns)
#             accs.append(data['test_acc'])
#             f1s.append(data['test_macro_avg_f1'])
#             # accepts.append(data['accept_score_correct_only'])

# dict_prepare = {

#     'target_dataset':targets,
#     'source_dataset':sources,
#     'n_shots':shots,
#     'accuracy':accs,
#     'F1_score':f1s,
#     # 'accept_score':accepts,
# }

# df = pd.DataFrame(dict_prepare, columns = ['target_dataset','source_dataset','n_shots',
#                                                'accuracy','F1_score'])


# df.to_json( '../result/zero-shot-slurm.json', orient='records', lines=True)
# # accept_scores = []
# # for d in dirs:
# #     accept_scores.append(read_accept(d))
# #     if read_accept(d)==None:
# #         print(d)
# # for path in dirs:
# #     if not os.path.exists(path +'/exp_correct_scores.json'):
# #         print(path)

# # for path in dirs:
# #     if os.path.exists(path +'/test_posthoc_analysis_1.txt'):
# #         print(path)


# import pandas as pd

# def extract_data_from_file(file_path):
#     # Reading the content of the provided file
#     with open(file_path, "r") as file:
#         content = file.readlines()

#     # Lists to store extracted data
#     hypotheses = []
#     premises = []
#     corrects = []
#     predicteds = []
#     considered_corrects = []

#     # Using a loop to handle potential inconsistencies in file structure
#     idx = 0
#     while idx < len(content):
#         line = content[idx]
#         if "Hypothesis:" in line:
#             hypotheses.append(line.replace("Hypothesis:", "").strip())
#             idx += 1
#         if "Premise:" in content[idx]:
#             premises.append(content[idx].replace("Premise:", "").strip())
#             idx += 1
#         if "Correct:" in content[idx]:
#             corrects.append(content[idx].replace("Correct:", "").strip().split('|')[0].strip())
#             idx += 1
#         if "Predicted:" in content[idx]:
#             predicteds.append(content[idx].replace("Predicted:", "").strip().split('|')[0].strip())
#             idx += 1
#         if "Considered Correct:" in content[idx]:
#             considered_corrects.append(True if "True" in content[idx] else False)
#             idx += 1
#         idx += 1  # Moving to the next block

#     # Creating a dataframe from extracted data
#     df = pd.DataFrame({
#         "Hypothesis:": hypotheses,
#         "Premise:": premises,
#         "Correct:": corrects,
#         "Predicted:": predicteds,
#         "Considered Correct:": considered_corrects
#     })

#     return df


# Dir = '/ukp-storage-1/jyang/few-shot-fact-checking/result/'
# target_datasets = ['sick', 'add_one','joci','mpe','dnc','hans','wnli','glue_diagnostics','conj',
#                    'snopes_stance','scifact','climate-fever-combined','vitaminc','covid_fact','fm2',
#                    'factcc','qags_cnndm','qags_xsum','xsum_hallucination']

# def merge_predictions(dataset_name):
#     tmp_dir = Dir + 'zero_shot/'+ dataset_name
#     df_accept = pd.read_json(tmp_dir+'/exp_correct_scores.json',lines=True)
    
#     filename = tmp_dir+"/test_posthoc_analysis.txt"

#     df = extract_data_from_file(filename)
#     df['Correct:'] = df['Correct:'].str.replace(" \|", "")
#     df = df.rename(columns={"Hypothesis:": "hypothesis", "Premise:": "premise"})
#     df['label'] = [row['Predicted:'].split(' | ')[0] for _, row in df.iterrows()]
#     try:
#         df['explanation'] = [row['Predicted:'].split(' | ')[1] for _, row in df.iterrows()]
#     except:
#         df['explanation'] = ""*len(df)
#     df = df.rename(columns={"Correct:":"gold", "label": "predicted_efever"})
#     df = df[['hypothesis', 'premise', 'gold', 'predicted_efever']]
#     df['premise'] = df['premise'].str.strip()
#     df['hypothesis'] = df['hypothesis'].str.strip()
#     df['gold'] = df['gold'].str.strip()
#     df['predicted_efever'] = df['predicted_efever'].str.strip()
    
#     df_all = df.merge(df_accept, on = ['hypothesis','premise'])
#     df_all = df_all.rename(columns={"accept_score": "acceptability_efever"})
#     df_all = df_all[['hypothesis','premise','gold','predicted_efever','acceptability_efever']]
#     df_all['target_dataset']=dataset_name
    
#     return df_all

# df_train = pd.DataFrame()
# for dataset in target_datasets:
#     print(dataset)
#     df_tmp = merge_predictions(dataset)
#     print(len(df_tmp))
#     df_train= pd.concat([df_train, df_tmp])


# df_train.to_json(Dir+'zero_shot_result.json', orient='records', lines=True)

import os
import json
import pandas as pd
# from nli_demo import length_mapping

# source_datasets=['efever']
# n_shots = [1,2,4,8,16,32,64,128]

# sample_selection = ['fastvotek','accept-fastvotek','least_confidence','ambiguous','accept-ambiguous']#,'ambiguous','accept-ambiguous'
# sub_set = [0,1,2,3,4]#
target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc',
                   'climate-fever-combined','snopes_stance','covid_fact',"scifact",
                   'factcc','xsum_hallucination','qags_xsum','qags_cnndm']

# target_datasets=['sick']
length_mapping = {
    'sick': 4906, 'add_one': 387, 'joci': 39092, 'mpe': 1000, 'dnc': 60036, 'hans': 30000,
    'wnli': 71, 'glue_diagnostics': 1104, 'conj': 623, 'wanli': 5000, 'robust_nli_3': 74922, 'scinli': 3000,
    'snopes_stance': 1651, 'scifact': 300, 'climate-fever-combined': 1381, 'vitaminc': 55197, 
    'covid_fact': 4086, 'dialfact': 11809, 'fm2': 1380, 'covert': 300,
    'factcc': 503, 'qags_cnndm': 714, 'qags_xsum': 239, 'xsum_hallucination': 1869, 'fib': 7158
}
source_datasets=['efever']
n_shots = [5000]
sample_selection = ['random']#,'ambiguous','accept-ambiguous'
sub_set = [0]#

# target_datasets =['wanli','scinli','robust_nli_3','dialfact','covert','fib']

def load_result(d):
    if os.path.exists(d +'/results_test.json'):
        f = open(d +'/results_test.json')
        data = json.load(f)
        return data
    else:
        print(d)

## Get more acc for ploting the trend
# sources=[]
# targets=[]
# selections=[]
# shots=[]
# subsets=[]
# results=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for selection in sample_selection:
#             for ns in n_shots: 
#                 for ss in sub_set:
#                     tmp={}
#                     path = '/'.join(('../result',
#                                             dataset,source,selection,'sub'+str(ss),'nt'+str(ns)))
#                     print(path)
#                     data=load_result(path)
#                     result = {}
#                     tmp['source_dataset']=source
#                     tmp['target_dataset']=dataset
#                     tmp['sample_selection']=selection
#                     tmp['n_shots']=ns
#                     tmp['sub_set']=ss
#                     tmp['acc00']=data['test_acc']
#                     tmp['acc30']=data['acc30']*100
#                     tmp['acc60']=data['acc60']*100
#                     tmp['acc90']=data['acc90']*100
#                     results.append(tmp)

# # print(results)
# with open("../result/acc_thresholds_accept_fastvotek_esnli.json", "w+") as f:
#     json.dump(results, f)


# results=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for selection in sample_selection:
#             for ns in n_shots: 
#                 for ss in sub_set:
#                     tmp={}
#                     path = '/'.join(('../result/zero_shot',dataset))
#                     print(path)
#                     data=load_result(path)
#                     result = {}
#                     tmp['source_dataset']='none'
#                     tmp['target_dataset']=dataset
#                     tmp['sample_selection']='none'
#                     tmp['n_shots']=0
#                     tmp['sub_set']=ss
#                     tmp['acc00']=data['test_acc']
#                     tmp['acc30']=0
#                     tmp['acc60']=0
#                     tmp['acc90']=0
#                     results.append(tmp)

# # print(results)
# with open("../result/acc_thresholds_zero.json", "w+") as f:
#     json.dump(results, f)

# dirs=[]
# sources=[]
# targets=[]
# selections=[]
# shots=[]
# subsets=[]
# accs=[]
# f1s=[]
# accepts=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for selection in sample_selection:
#             for ns in n_shots: 
#                 for ss in sub_set:
#                     path = '/'.join(('../result',
#                                             dataset,source,selection,'sub'+str(ss),'nt'+str(ns)))
# #                     print(path)
#                     df_data=pd.read_json(path+'/exp_correct_scores.json',lines=True)
#                     acceptability_correct = (df_data['accept_score'].sum())/length_mapping[dataset]
#                     acc90 = len(df_data.loc[df_data['accept_score']>=0.90]['accept_score'])/length_mapping[dataset]
#                     acc60 = len(df_data.loc[df_data['accept_score']>=0.60]['accept_score'])/length_mapping[dataset]
#                     acc30 = len(df_data.loc[df_data['accept_score']>=0.30]['accept_score'])/length_mapping[dataset]
                    
#                     if os.path.exists(path +'/results_test.json'):
#                         f = open(path +'/results_test.json')
#                         result = json.load(f)
#                         result['avg_acceptability_correct']=acceptability_correct
#                         result['acc90']=acc90
#                         result['acc60']=acc60
#                         result['acc30']=acc30

#                         with open(path +'/results_test.json', 'w') as f:
#                             json.dump(result, f)

# dirs=[]
# sources=[]
# targets=[]
# selections=[]
# shots=[]
# subsets=[]
# accs=[]
# f1s=[]
# # accepts=[]
# acceptability_corrects=[]
# acc90s=[]
# acc60s=[]
# acc30s=[]
# for source in source_datasets:
#     for dataset in target_datasets:
#         for selection in sample_selection:
#             for ns in n_shots: 
#                 for ss in sub_set:
#                     path = '/'.join(('../result',
#                                             dataset,source,selection,'sub'+str(ss),'nt'+str(ns)))

#                     data=load_result(path)
#                     try:
#                         dirs.append(path)
#                         sources.append(source)
#                         targets.append(dataset)
#                         selections.append(selection)
#                         shots.append(ns)
#                         subsets.append(ss)
#                         accs.append(data['test_acc'])
#                         f1s.append(data['test_macro_avg_f1'])
#                         accepts.append(data['accept_score_correct_only'])
#                         acceptability_corrects.append(data['avg_acceptability_correct'])
#                         acc90s.append(data['acc90'])
#                         acc60s.append(data['acc60'])
#                         acc30s.append(data['acc30'])
#                     except Exception as e: 
#                         print(e)
#                         print(path)

# dict_prepare = {

#     'target_dataset':targets,
#     'source_dataset':sources,
#     'sample_selection':selections,
#     'n_shots':shots,
#     'sub_set':subsets,
#     'accuracy':accs,
#     'F1_score':f1s,
#     'accept_score':accepts,
#     'avg_acceptability_correct':acceptability_corrects,
#     'acc90':acc90s,
#     'acc60':acc60s,
#     'acc30':acc30s,
# }

# df = pd.DataFrame(dict_prepare, columns = ['target_dataset','source_dataset','sample_selection', 'n_shots',
#                                                'sub_set','accuracy','F1_score','accept_score',
#                                                'avg_acceptability_correct','acc90','acc60','acc30'])


# df.to_json( '../result/full-shot-efever-slurm.json', orient='records', lines=True)

## Get more acc for ploting the trend

results=[]
for source in source_datasets:
    for dataset in target_datasets:
        for selection in sample_selection:
            for ns in n_shots: 
                for ss in sub_set:
                    tmp={}
                    path = '/'.join(('../result',
                                            dataset,source,selection,'sub'+str(ss),'nt'+str(ns)))
                    df_data=pd.read_json(path+'/exp_correct_scores.json',lines=True)
                    result = {}
                    for thre in range(0,100,10):
#                     acceptability_correct = (df_data['accept_score'].sum())/length_mapping[dataset]
                        acc = len(df_data.loc[df_data['accept_score']>=thre/100]['accept_score'])/length_mapping[dataset]
                        result['acc'+str(thre)]=acc

                    with open(path +'/acc_threshold_test.json', 'w') as f:
                        json.dump(result, f)
#                     if source=='efever':
                    tmp['source_dataset']=source
                    tmp['target_dataset']=dataset
                    tmp['sample_selection']=selection
                    tmp['n_shots']=ns
                    tmp['sub_set']=ss
                    tmp['acc_thres']=result
                    results.append(tmp)

with open("../result/acc_thresholds_slurm_efever.json", "w+") as f:
    json.dump(results, f)