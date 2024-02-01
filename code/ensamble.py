import pandas as pd
import os
import re
# from utils import sep_label_explanation
# from nli_demo import extract_data_from_file
from load_target_dataset import load_raw_data
from feature_conversion_methods import label_mapping
label2text = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
# target_datasets = ['dnc']
def most_common(lst):
    return max(set(lst), key=lst.count)

def extract_data_from_file(file_path):
    data = {
        "Hypothesis": [],
        "Premise": [],
        "Correct": [],
        "Predicted": [],
        "Considered Correct": []
    }

    # Initialize a temporary dictionary to hold the current set of data
    temp_data = {key: "" for key in data}

    # Regular expression pattern to match the labels
    pattern = re.compile(r"^(Hypothesis|Premise|Correct|Predicted|Considered Correct):")

    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line starts with one of our labels
            match = pattern.match(line)
            if match:
                # If we have collected text for a previous label, add it to the corresponding list
                current_label = match.group(1)
                if temp_data[current_label]:  # If there's already text for this label, it means we've reached a new record
                    for label, text in temp_data.items():
                        data[label].append(text)
                    temp_data = {key: "" for key in data}  # Reset for the next record
                # Start collecting text for the new label
                temp_data[current_label] = line[len(match.group(0)):].strip()
            elif current_label:
                # This is a continuation of the text for the current label
                temp_data[current_label] += " " + line.strip()
        
        # Don't forget to save the last record after the loop ends
        for label, text in temp_data.items():
            data[label].append(text)

    # Remove the 'Correct' key if it has only empty strings, as we didn't find this label in the unique lines
    if all(len(text.strip()) == 0 for text in data["Correct"]):
        data.pop("Correct")

    # Return the data with potential empty strings where sections were missing
    df_data = pd.DataFrame(data)
    
    return df_data

def sep_label_explanation(lines, explanation_sep):
    
    # broken_generation = 0
    labels = []
    explanations = []
    for line in lines:
        line_split = line.split(explanation_sep)
        if len(line_split) > 1:
            labels.append(line_split[0].strip()) #text label: entailment, neutral, contradiction
            explanations.append(line_split[1].strip())
        else: 
            # broken_generation+=1
            try:
                # print(f"This line couldn't be processed (most likely due to format issue): {line}")
                labels.append(line.split()[0]) #text label: maybe nonsense, we assume the first word is always the label
                explanations.append(' '.join(line.split()[1:]))
            except:
                labels.append(line) #the line is totally empty
                explanations.append('')  

    # l_pred = [l.split(explanation_sep)[0].strip() for l in predictions]
    # print("predictions:",l_pred)
    # e_pred = 
    # references = sum(references, [])  
    # l_true = [l.split(explanation_sep)[0].strip() for l in references]
    # print("predictions:",l_true)

    # acc = [l_pred[i] == l_true[i] for i in range(len(predictions))]
    return labels, explanations

def get_label_ensamble(dataset, n_shots):
    # labels = []
    df_label_ensamble = pd.DataFrame()
    for source in source_dataset:
        for ss in subset:
            for sslect in sample_selection:
                result_path = '/'.join(('../result', dataset, source, sslect,'sub'+str(ss),'nt'+str(n_shots)))

                if os.path.exists(result_path+"/test_posthoc_analysis_1.txt"):
                    filename = result_path+"/test_posthoc_analysis_1.txt"
                else:
                    filename = result_path+"/test_posthoc_analysis.txt"  

                df = extract_data_from_file(filename)
                df = df.rename(columns={"Hypothesis": "hypothesis", "Premise": "premise"})
                df['label'] = [row['Predicted'].split(' ')[0] for _, row in df.iterrows()]
                df_label_ensamble[result_path]=df['label']
#     df_label_ensamble['majority_vote'] = df_label_ensamble.apply(lambda row: most_common(list(row)), axis=1)
    true_label= [row['Correct'].split(' ')[0] for _, row in df.iterrows()]
    df_label_ensamble['true_label']=true_label
#     if len(set(true_label))==2:
#         df_label_ensamble['majority_vote'] = df_label_ensamble['majority_vote'].replace({'contradiction': 'not_entailment', 'neutral': 'not_entailment'})
#     # Manual calculation of accuracy
    return df_label_ensamble

def acc_ensamble(dataset, n_shots):
    # labels = []
    # Manual calculation of accuracy
    df_label_ensamble = get_label_ensamble(dataset,n_shots)
#     true_label = list()
    correct_predictions = sum(majority_vote == truth for majority_vote, truth in zip(df_label_ensamble['majority_vote'], df_label_ensamble['true_label']))
    accuracy = correct_predictions / len(df_label_ensamble['true_label'])

    print(f"For {dataset}, the accuracy of the majority vote is: {accuracy:.4f}")
    return accuracy

def average_prob(dataset, df):
    df_label_ensamble = pd.DataFrame()
    
    df_full = df.loc[df['dataset']==dataset]
    columns=list(df_full.columns)
    
    columns_containing_en = [col for col in columns if '_en' in col]
    columns_containing_neutral = [col for col in columns if '_neutral' in col]
    columns_containing_contradiction = [col for col in columns if '_contradiction' in col]

    df_label_ensamble['prob_en_avg']=df_full[columns_containing_en].mean(axis=1)
    df_label_ensamble['prob_neutral_avg']=df_full[columns_containing_neutral].mean(axis=1)
    df_label_ensamble['prob_contradiction_avg']=df_full[columns_containing_contradiction].mean(axis=1)
    df_label_ensamble['true_label']=df_full['gold_label']
       
    df_label_ensamble['predicted_label'] = df_label_ensamble[['prob_en_avg', 'prob_neutral_avg', 'prob_contradiction_avg']].idxmax(axis=1)

    # Map the column names to the actual labels.
    local_label_mapping = {
        'prob_en_avg': 'entailment',
        'prob_neutral_avg': 'neutral',
        'prob_contradiction_avg': 'contradiction'
    }

    # Apply the mapping to the 'predicted_label' to replace column names with actual labels.
    df_label_ensamble['predicted_label'] = df_label_ensamble['predicted_label'].map(local_label_mapping)

    if len(set(df_full['gold_label']))==2: # convert predicted label from 3 classes to 2 classes
        df_label_ensamble['predicted_label'] = df_label_ensamble['predicted_label'].replace({'contradiction': 'not_entailment', 'neutral': 'not_entailment'})
         
    correct_predictions = sum(majority_vote == truth for majority_vote, truth in zip(df_label_ensamble['predicted_label'], df_label_ensamble['true_label']))
    accuracy = correct_predictions / len(df_label_ensamble['true_label'])
    print(f"For {dataset}, the accuracy of the majority vote compared to the ground truth is: {accuracy:.4f}")
    
    return accuracy

def min_prob(dataset, df):
    df_label_ensamble = pd.DataFrame()
    
    df_full = df.loc[df['dataset']==dataset]
    columns=list(df_full.columns)
    
    columns_containing_en = [col for col in columns if '_en' in col]
    columns_containing_neutral = [col for col in columns if '_neutral' in col]
    columns_containing_contradiction = [col for col in columns if '_contradiction' in col]

    df_label_ensamble['prob_en_avg']=df_full[columns_containing_en].min(axis=1)
    df_label_ensamble['prob_neutral_avg']=df_full[columns_containing_neutral].min(axis=1)
    df_label_ensamble['prob_contradiction_avg']=df_full[columns_containing_contradiction].max(axis=1)
    df_label_ensamble['true_label']=df_full['gold_label']
       
    df_label_ensamble['predicted_label'] = df_label_ensamble[['prob_en_avg', 'prob_neutral_avg', 'prob_contradiction_avg']].idxmax(axis=1)

    # Map the column names to the actual labels.
    local_label_mapping = {
        'prob_en_avg': 'entailment',
        'prob_neutral_avg': 'neutral',
        'prob_contradiction_avg': 'contradiction'
    }

    # Apply the mapping to the 'predicted_label' to replace column names with actual labels.
    df_label_ensamble['predicted_label'] = df_label_ensamble['predicted_label'].map(local_label_mapping)

    if len(set(df_full['gold_label']))==2: # convert predicted label from 3 classes to 2 classes
        df_label_ensamble['predicted_label'] = df_label_ensamble['predicted_label'].replace({'contradiction': 'not_entailment', 'neutral': 'not_entailment'})
         
    correct_predictions = sum(majority_vote == truth for majority_vote, truth in zip(df_label_ensamble['predicted_label'], df_label_ensamble['true_label']))
    accuracy = correct_predictions / len(df_label_ensamble['true_label'])
    print(f"For {dataset}, the accuracy of the majority vote compared to the ground truth is: {accuracy:.4f}")
    
    return accuracy
# acc_avg = 0
# for dataset in target_datasets:
#     acc_avg+=acc_ensamble(dataset,64)

# acc_avg = acc_avg/len(target_datasets)
# # print(f"For all dataset, the average accuracy is: {acc_avg:.4f}")
# for dataset in target_datasets:
#     get_label_ensamble(dataset,64).to_json( '../result/ensamble/slurm_ensamble_efever_'+dataset+'.json', orient='records', lines=True)

source_dataset = ['esnli','efever']
sample_selection = ['accept-fastvotek'] #,'ambiguous','accept-ambiguous'
n_shots = 64
subset = [0,1,2,3,4]

target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc',
                   'climate-fever-combined','snopes_stance','covid_fact',"scifact",
                   'factcc','xsum_hallucination','qags_xsum','qags_cnndm']

df=pd.DataFrame()
for dataset in target_datasets:
    df_dataset=pd.DataFrame()
    data=load_raw_data(dataset)
    gold_label=data['label']
    for source in source_dataset:
        for ss in subset:
            for sslect in sample_selection:
                result_path = '/'.join(('../result', dataset, source, sslect,'sub'+str(ss),'nt'+str(n_shots)))
                model_name='/'.join((source, sslect,'sub'+str(ss),'nt'+str(n_shots)))
                df_tmp = pd.read_json(result_path+'/l_e_prob.json',lines=True)
                df_dataset[model_name+'_en']=df_tmp['prob_en']
                df_dataset[model_name+'_neutral']=df_tmp['prob_neutral']
                df_dataset[model_name+'_contradiction']=df_tmp['prob_contradiction']
                df_dataset['dataset']=dataset
                if dataset in label_mapping.keys():
                    df_dataset['gold_label']=[label_mapping[dataset][gold_l] for gold_l in gold_label]
                else:
                    df_dataset['gold_label'] = gold_label
                df_dataset[model_name+'_label']=[label2text[pred_l] for pred_l in list(df_tmp['label'])]

    df=pd.concat([df,df_dataset])

acc_avg = 0
for dataset in target_datasets:
    acc=average_prob(dataset,df)
    acc_avg+=acc

acc_avg = acc_avg/len(target_datasets)
print(f"For all dataset, the average accuracy is: {acc_avg:.4f}")