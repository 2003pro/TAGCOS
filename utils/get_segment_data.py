import pandas as pd
import json

input_path = 'out/top_omp_LESS102030_100.pkl'
data = pd.read_pickle(input_path)
datasets_to_combine = [
    'science.scifact_json',
    'science.evidence_inference',
    'science.evidence_inference_data',
    'science.qasper_truncated_4000',
    'science.scierc_ner',
    'science.scierc_relation',
    'science.scitldr_aic'
]
data['dataset'] = data['dataset'].replace(datasets_to_combine, 'science')
grouped = data.groupby('dataset')['id'].apply(list)
output_path = 'out/top_omp_LESS102030_100_data.jsonl'
#import pdb;pdb.set_trace()
with open(output_path, 'w', encoding='utf-8') as outfile:
    for dataset, ids in grouped.items():
        dataset_process = dataset.replace('selected_full_data_102030/', '')
        file_path = f'data/train/processed/{dataset_process}/{dataset_process}_data.jsonl'
        with open(file_path, 'r', encoding='utf-8') as infile:
            for index, line in enumerate(infile):
                if index in ids:
                    outfile.write(line)

print(f"Data has been written to {output_path}")
