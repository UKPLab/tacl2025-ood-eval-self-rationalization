from collections import defaultdict
import random
"""
Example-to-Feature conversion methods
Modified from
https://github.com/salesforce/cos-e/blob/master/code/generation/train_commonsenseqa_v1.0.py and ""_v1.11.py (identical)
as well as Tensorflow code for WTF?: 
https://github.com/google-research/google-research/blob/master/wt5/wt5/preprocessors.py
"""
# This code is based on https://github.com/allenai/label_rationale_association/blob/main/feature_conversion_methods.py

# wt5_esnli_label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'} 
# copa_label_mapping = {0: 'not entailment', 1: 'entailment'} 

label_mapping = {
    "esnli": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "anli": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "efever": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'joci': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'wanli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'conj': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'scinli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'sick': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    # 'snli_hard': {0: 'entailment', 1: 'neutral', 2: 'contradiction'}, This dataset is included in robust_nli
    'robust_nli_3': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mpe': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'glue_diagnostics': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "copa": {0: 'not_entailment', 1: 'entailment'},
    "abductive_nli": {0: 'not_entailment', 1: 'entailment'}, 
    "wnli": {0: 'not_entailment', 1: 'entailment'},
    "add_one": {0: 'not_entailment', 1: 'entailment'},
    "dnc": {0: 'not_entailment', 1: 'entailment'},
    "iie": {0: 'not_entailment', 1: 'entailment'},
    "docnli": {0: 'not_entailment', 1: 'entailment'},
    "scitail": {0: 'not_entailment', 1: 'entailment'},
    "hans": {0: 'entailment', 1: 'not_entailment'}, # the label is opposite of the others
    "fm2": {0: 'entailment', 2: 'not_entailment'},
    "covid_fact": {0: 'entailment', 2: 'not_entailment'},
    'covert':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'vitaminc':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'snopes_stance':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-combined':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-separate':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'dialfact-no-context':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'dialfact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'scifact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'factcc':{0: 'entailment', 1: 'not_entailment'},
    'qags_cnndm':{0: 'entailment', 1: 'not_entailment'},
    'qags_xsum':{0: 'entailment', 1: 'not_entailment'},
    'xsum_hallucination':{0: 'entailment', 1: 'not_entailment'},
    'fib':{0: 'entailment', 1: 'not_entailment'},
    'frank':{0: 'not_entailment', 1: 'entailment'}
}

def format_instance(
        example,
        task,
        tokenizer,
        explanation_sep,
        max_seq_length=None,
        io_format=None, 
):

    input_string, answer_string = formatting(example, task, io_format, explanation_sep)

    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())

    # input_string = ' '.join(input_string.split())
    # answer_string = ' '.join(answer_string.split())

    encodings = tokenizer.encode_plus(
        input_string,
        max_length=max_seq_length,
        pad_to_max_length=False,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    # note even with "lm_labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string,
        max_length=max_seq_length,
        pad_to_max_length=False,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]
    encodings["question_encoding"] = encodings["input_ids"]

    #return encodings
    return {**example, **encodings}


def formatting(item, dataset_name, io_format='standard', explanation_sep=' "explanation: " '):
    # print(item)
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
    
    return input_string, answer_string