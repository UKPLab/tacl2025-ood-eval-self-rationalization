import datasets
import pandas as pd
import numpy as np
from feature_conversion_methods import formatting
from save_val_dataset import random_select
import re
URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')

inverse_label_mapping = {
    "esnli": {'entailment':0, 'neutral':1, 'contradiction':2},
    'joci': {'entailment':0, 'neutral':1, 'contradiction':2},
    'sick': {'entailment':0, 'neutral':1, 'contradiction':2},
    'scinli': {'entailment':0, 'neutral':1, 'contrasting':2},
    'wanli': {'entailment':0, 'neutral':1, 'contradiction':2},
    'robust_nli_3': {'entailment':0, 'neutral':1, 'contradiction':2},
    'glue_diagnostics': {'entailment':0, 'neutral':1, 'contradiction':2},
    'docnli': {'not_entailment':0, 'entailment':1},
    "copa": {'not_entailment':0, 'entailment':1},
    "wnli": {'not_entailment':0, 'entailment':1},
    "iie": {'not_entailment':0, 'entailment':1},
    "copa": {'not_entailment':0, 'entailment':1},
    "scitail": {'not_entailment':0, 'entailment':1},
    "robust_IS_SD": {'non-entailment':0, 'entailment':1},
    "robust_LI_TS": {'non-contradiction':0, 'contradiction':1},
    "fm2": {'SUPPORTS':0, 'REFUTES':2},
    "covid_fact": {'SUPPORTED':0, 'REFUTED':2},
    "covert": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "vitaminc": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "efever": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "snopes_stance": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},  
    "scifact": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "wice": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "climate-fever-combined": {'SUPPORTS':0, 'NOT_ENOUGH_INFO':1, 'REFUTES':2},
    "climate-fever-separate": {'SUPPORTS':0, 'NOT_ENOUGH_INFO':1, 'REFUTES':2}, 
    "dialfact-no-context": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "dialfact": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},  
    "factcc": {'CORRECT':0, 'INCORRECT':1},
    "qags_cnndm": {'CORRECT':0, 'INCORRECT':1},
    "qags_xsum": {'CORRECT':0, 'INCORRECT':1},
    "xsum_hallucination": {'CORRECT':0, 'INCORRECT':1},
    "fib": {'CORRECT':0, 'INCORRECT':1},
}

def load_raw_data(dataset_name, split='test'):

    if dataset_name in ['esnli', 'anli', 'efever'] and split =='val':
        data = datasets.load_dataset('json', data_files='../datasets/source/'+dataset_name+'/val_select.json', split='train')

    if dataset_name =='esnli' and split =='train':
        data = datasets.load_dataset(dataset_name, split='train')
    
    if dataset_name =='esnli' and split =='human_exp':
        data = datasets.load_dataset('csv', data_files='../datasets/source/'+dataset_name+'/human_explanations.csv', split='train')
        data = data.rename_column("answer", "label")
        data = data.rename_column("our_explanation", "explanation_1")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)
        data = data.shuffle(seed=42)
        data = random_select(data, 32)

    if dataset_name =='esnli' and split =='orig_exp':
        data = datasets.load_dataset('csv', data_files='../datasets/source/'+dataset_name+'/human_explanations.csv', split='train')
        data = data.rename_column("answer", "label")
        data = data.rename_column("orig_explanation", "explanation_1")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)
        data = data.shuffle(seed=42)
        data = random_select(data, 32)

    if dataset_name =='anli' and split =='train':
        data = datasets.load_dataset(dataset_name)
        data = datasets.concatenate_datasets([data["dev_r1"], data["dev_r2"], data["dev_r3"]]) # we use dev because train set doesn't have explanations
        data = data.rename_column("reason", "explanation_1")

    if dataset_name == 'anli' and split == 'test':
        data = datasets.load_dataset(dataset_name)
        data = datasets.concatenate_datasets([data["test_r1"], data["test_r2"], data["test_r3"]])
        data = data.rename_column("reason", "explanation_1")   

    if dataset_name =='efever' and split =='train':
        fever = pd.read_json('../datasets/source/efever/fever_train.jsonl', lines=True)
        efever = pd.read_json('../datasets/source/efever/efever_train_set.jsonl', lines=True)
        df_data = pd.merge(fever, efever, on='id', how='inner')
        df_data = df_data.drop(columns=['id', 'verifiable', 'evidence'])
        df_data = df_data.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis", 'summary': 'explanation_1'})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
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
          
    if dataset_name in ['copa','wnli', 'iie', 'scitail']:
        data = datasets.load_dataset('csv', data_files='../datasets/target/nli/NLI_datasets/nli_datasets/'+dataset_name+'/dev.csv', split='train')
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name =='sick':
        data = datasets.load_dataset(dataset_name)
        data = data['test']

        cols_to_remove = data.column_names
        cols_to_remove.remove("sentence_A")
        cols_to_remove.remove("sentence_B")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("sentence_A", "premise")
        data = data.rename_column("sentence_B", "hypothesis")
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)  

    if dataset_name =='joci':
        data = datasets.load_dataset('pietrolesci/joci')
        data = data['full']
        cols_to_remove = data.column_names
        cols_to_remove.remove("context")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("context", "premise")
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)   

    if dataset_name =='mpe':
        data = datasets.load_dataset('pietrolesci/mpe')
        data = data['test']
        cols_to_remove = data.column_names
        cols_to_remove.remove("premise")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)  
    
    if dataset_name =='dnc':
        data = datasets.load_dataset('pietrolesci/dnc')
        data = data['test']
        cols_to_remove = data.column_names
        cols_to_remove.remove("context")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("context", "premise")
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name =='hans':
        data = datasets.load_dataset(dataset_name)
        data = data['validation']
        cols_to_remove = data.column_names
        cols_to_remove.remove("premise")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)             

    if dataset_name =='conj':
        data = datasets.load_dataset('pietrolesci/conj_nli')
        data = data['dev']

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)  
    
    if dataset_name =='add_one':
        data = datasets.load_dataset('pietrolesci/add_one_rte')
        data = data['test']
        
        cols_to_remove = data.column_names
        cols_to_remove.remove("premise")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)   

    if dataset_name =='wanli':
        data = datasets.load_dataset('alisawuffles/WANLI')
        data = data['test']
        
        cols_to_remove = data.column_names
        cols_to_remove.remove("premise")
        cols_to_remove.remove("hypothesis")
        cols_to_remove.remove("gold")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("gold", "label")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)   

    if dataset_name =='docnli':
        data = datasets.load_dataset('saattrupdan/doc-nli')
        data = data['test']
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
    
    if dataset_name =='robust_nli_3':
        data = datasets.load_dataset('json', data_files='../datasets/target/nli/robust_nli/robust_nli.txt', split='train') 
        data = data.rename_column("prem", "premise")
        data = data.rename_column("hypo", "hypothesis")

        data = data.filter(lambda example: example['split'] != 'IS_SD')
        data = data.filter(lambda example: example['split'] != 'LI_TS')

        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'scinli':
        data = datasets.load_dataset('metaeval/scinli')
        data = data['test']
        data = data.rename_column("sentence1", "premise")
        data = data.rename_column("sentence2", "hypothesis")
        data = data.filter(lambda example: example["label"]!='reasoning')

        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
        
    if dataset_name=='fm2':
        data = datasets.load_dataset('json', data_files='../datasets/target/fact-check/fm2/test.jsonl', split='train')
        df_data = data.to_pandas()
        evidence = [' '.join([e['text'] for e in row['gold_evidence']]) for _, row in df_data.iterrows()]
        data = data.add_column("evidence", evidence) 

        cols_to_remove = data.column_names
        cols_to_remove.remove("text") # text is claim
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("text", "hypothesis")
        data = data.rename_column("evidence", "premise")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name=='vitaminc':
        data = datasets.load_dataset('tals/vitaminc')
        data = data['test']

        cols_to_remove = data.column_names
        cols_to_remove.remove("claim")
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("evidence", "premise")

        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)             

    if dataset_name == 'glue_diagnostics':
        data = datasets.load_dataset('csv', data_files='../datasets/target/nli/glue_diagnostics/diagnostic-full.tsv', delimiter='\t', split='train')
        cols_to_remove = data.column_names
        cols_to_remove.remove("Premise")
        cols_to_remove.remove("Hypothesis")
        cols_to_remove.remove("Label")
        data = data.remove_columns(cols_to_remove)

        data = data.rename_column("Premise", "premise")
        data = data.rename_column("Hypothesis", "hypothesis")
        data = data.rename_column("Label", "label")

        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
    
    if dataset_name == 'abductive_nli':
        data = datasets.load_dataset('json', data_files='../datasets/target/nli/abductive_nli/anli/test.jsonl',split='train')
        df_data = data.to_pandas()
        with open('../datasets/target/nli/abductive_nli/anli/test-labels.lst', 'r') as file:
            labels = np.array([line.strip() for line in file])

        # one-hot encoding
        labels = labels.astype(int)-1 
        one_hot_labels = np.zeros((len(labels), 2))
        one_hot_labels[np.arange(len(labels)), labels] = 1

        df_data['label_h1'] = one_hot_labels[:,0]
        df_data['label_h2'] = one_hot_labels[:,1]
        df_data['premise'] = [' '.join((row['obs1'], row['obs2'])) for _, row in df_data.iterrows()]
        del df_data['story_id']
        df_data_1 = df_data[['premise', 'hyp1', 'label_h1']]
        df_data_1 = df_data_1.rename(columns={"hyp1": "hypothesis", "label_h1": "label"})

        df_data_2 = df_data[['premise', 'hyp2', 'label_h2']]
        df_data_2 = df_data_2.rename(columns={"hyp2": "hypothesis", "label_h2": "label"})

        df_data = pd.concat([df_data_1, df_data_2])
        df_data['label'] = df_data['label'].astype(int)
        # df_data.to_json('../datasets/target/nli/abductive_nli/anli/processed_test.json', orient='records', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames) 

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)
        
    if dataset_name == 'covid_fact':

        data = datasets.load_dataset('json', data_files='../datasets/target/fact-check/covid_fact/COVIDFACT_dataset.jsonl',split='train')

        cols_to_remove = data.column_names
        cols_to_remove.remove("claim") 
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("evidence", "premise")

        data = data.map(lambda example: {'premise': " ".join(example['premise'])})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'covert':
        df_data = pd.read_json('../datasets/target/fact-check/covert/CoVERT_FC_annotations.jsonl', lines=True)
        claim = [re.sub(URL, 'URL', row['claim']) for _, row in df_data.iterrows()]
        claim = [re.sub('@username', '', c) for c in claim]
        evidence = [' '.join([e[2] for e in row['evidence'] if e[2] is not None]) for _, row in df_data.iterrows()]
        df_data['claim'] = claim
        df_data['evidence'] = evidence
        df_data = df_data.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'snopes_stance':

        df_data = pd.read_json('../datasets/target/fact-check/snopes/ukp_snopes_corpus/datasets/snopes.stance.test.jsonl', lines=True)
        df_evidence = pd.read_json('../datasets/target/fact-check/snopes/ukp_snopes_corpus/datasets/snopes.page.json').T 
        df_data['evidence'] = [" ".join([df_evidence.loc[evi[0]]['lines'][evi[1]] for evi in row['evidence'][0]]) for _, row in df_data.iterrows()]

        df_data = df_data.drop(columns=['id', 'verifiable', 'predicted_evidence', 'predicted_pages'])
        df_data = df_data.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'scifact':
        df_data = pd.read_json('../datasets/target/fact-check/scifact/data/claims_dev.jsonl', lines=True) 
        df_evidence = pd.read_json('../datasets/target/fact-check/scifact/data/corpus.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            claim.append(row['claim'])
            evi = []
            for e in row['cited_doc_ids']:
                df = df_evidence.loc[df_evidence['doc_id']==e]
                evi.append(' '.join(df.iloc[0]['abstract']))
            evidence.append(' '.join(evi))  
            if not row['evidence']:
                label.append('NOT ENOUGH INFO')
            else:
                l = 0
                s = 0
                for _, value in enumerate(row['evidence'].values()):
                    s = s+len(value)
                    for v in value:
                        if v['label']=='CONTRADICT':
                            l+=1
                if l/s<0.5:
                    label.append('SUPPORTS')
                else:
                    label.append('REFUTES')
                    
        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])

        df_data['explanation_1'] = [""]*len(df_data)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)    

    if dataset_name == 'wice':
        data = datasets.load_dataset('json', data_files='../datasets/target/fact-check/wice/claim/test.jsonl', split='train') 

        data = data.map(lambda example: {'premise': " ".join([example['evidence'][e] for e in example['supporting_sentences'][0]])})
        cols_to_remove = data.column_names
        cols_to_remove.remove("claim") 
        cols_to_remove.remove("premise")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'climate-fever-combined':
        data=datasets.load_dataset('json', data_files='../datasets/target/fact-check/climate-fever/climate-fever-dataset-r1.jsonl')
        data=data['train']

        cols_to_remove = data.column_names
        cols_to_remove.remove("claim")
        cols_to_remove.remove("evidences")
        cols_to_remove.remove("claim_label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("evidences", "premise")
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("claim_label", "label")
        data = data.filter(lambda example: example["label"]!='DISPUTED')

        data = data.map(lambda example: {'premise': " ".join([x['evidence'] for x in example['premise']])})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'climate-fever-separate':
        df_data = pd.read_json('../datasets/target/fact-check/climate-fever/climate-fever-dataset-r1.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            for e in row['evidences']:
                claim.append(row['claim'])
                evidence.append(e['evidence'])
                label.append(e['evidence_label'])
        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)    
    
    if dataset_name == 'dialfact-no-context': # context are not combined with evidence
        data = datasets.load_dataset('json', data_files='../datasets/target/fact-check/dialfact/test_split.jsonl', split='train') 
        cols_to_remove = data.column_names
        cols_to_remove.remove("context") 
        cols_to_remove.remove("response") # response is the claim
        cols_to_remove.remove("evidence_list")
        cols_to_remove.remove("response_label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("response", "hypothesis")
        data = data.rename_column("evidence_list", "premise")
        data = data.rename_column("response_label", "label")

        data = data.map(lambda example: {'premise': " ".join([x[2] for x in example['premise']])})
        data = data.map(lambda example: {'context': " ".join(example['context'])})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'dialfact': # context are not combined with evidence
        data = datasets.load_dataset('json', data_files='../datasets/target/fact-check/dialfact/test_split.jsonl', split='train') 
        cols_to_remove = data.column_names
        cols_to_remove.remove("context") 
        cols_to_remove.remove("response") # response is the claim
        cols_to_remove.remove("evidence_list")
        cols_to_remove.remove("response_label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("response", "hypothesis")
        data = data.rename_column("evidence_list", "premise")
        data = data.rename_column("response_label", "label")

        data = data.map(lambda example: {'premise': " ".join([x[2] for x in example['premise']])})
        data = data.map(lambda example: {'context': " ".join(example['context'])})
        data = data.map(lambda example: {'premise': " ".join((example['context'],example['premise']))})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'factcc':
        data = datasets.load_dataset('json', data_files='../datasets/target/hallucination/factcc/annotated_data/test/data-dev.jsonl', split='train') 
        cols_to_remove = data.column_names
        cols_to_remove.remove("claim") # text is claim
        cols_to_remove.remove("label")
        cols_to_remove.remove("text")
        # cols_to_remove.remove("response_label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("text", "premise")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'qags_cnndm':
        df_data = pd.read_json('../datasets/target/hallucination/qags/mturk_cnndm.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            for e in row['summary_sentences']:
                claim.append(e['sentence'])
                evidence.append(row['article'])
                y=0
                for r in e['responses']:
                    if r['response']=='yes':
                        y+=1
                if y>=2:
                    label.append('CORRECT')
                else:
                    label.append('INCORRECT')

        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)    
        df_data.to_json('../datasets/target/hallucination/qags/processed_cnndm.json', orient='records', lines=True)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "not_entailment"])
        data = data.cast_column("label", labelNames)

    if dataset_name == 'qags_xsum':
        df_data = pd.read_json('../datasets/target/hallucination/qags/mturk_xsum.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            for e in row['summary_sentences']:
                claim.append(e['sentence'])
                evidence.append(row['article'])
                y=0
                for r in e['responses']:
                    if r['response']=='yes':
                        y+=1
                if y>=2:
                    label.append('CORRECT')
                else:
                    label.append('INCORRECT')

        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)    
        df_data.to_json('../datasets/target/hallucination/qags/processed_xsum.json', orient='records', lines=True)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "not_entailment"])
        data = data.cast_column("label", labelNames)

    if dataset_name == 'xsum_hallucination':
        df_data = pd.read_csv('../datasets/target/hallucination/xsum_hallucination/factuality_annotations_xsum_summaries.csv')
        grouped_df = df_data.groupby(['bbcid','system'])
        df_aggre = grouped_df.aggregate(lambda x: tuple(x))
        data_bbc = datasets.load_dataset('xsum', split='test')
        df_bbc = data_bbc.to_pandas()
        claim = []
        evidence_id = []
        label = []
        for index, row in df_aggre.iterrows():
            claim.append(row['summary'][0])
            evidence_id.append(str(index[0]))
            l=0
            for y in row['is_factual']:
                if y=='yes':
                    l+=1 
            if l>=2:
                label.append('CORRECT')
            else:
                label.append('INCORRECT')
        dict_prepare = {
            'hypothesis':claim,
            'premise_id':evidence_id,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise_id', 'label'])
        merged_df = pd.merge(df_data, df_bbc, left_on='premise_id', right_on='id', how='left')
        merged_df = merged_df.drop('id', axis=1)
        data = datasets.Dataset.from_pandas(merged_df)
        # print(data)
        cols_to_remove = data.column_names
        cols_to_remove.remove("hypothesis") 
        cols_to_remove.remove("document") 
        cols_to_remove.remove("label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("document", "premise")
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})   

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'fib':

        df_data = pd.read_json('../datasets/target/hallucination/fib/fib.json',lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            claim.append(row['correct_choice'])
            evidence.append(row['input'])
            label.append('CORRECT')

            claim.append(row['list_choices'][1-int(row['lbl'])])
            evidence.append(row['input'])
            label.append('INCORRECT')

        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "not_entailment"])
        data = data.cast_column("label", labelNames)

    return data

def load_format_data(dataset_name, split='test'):

    data = load_raw_data(dataset_name, split)
    input_string, answer_string = zip(*list(map(lambda x: formatting(x, dataset_name), data)))
    data = data.add_column("input_string", input_string)
    data = data.add_column("answer_string", answer_string)

    return data