import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np

def custom_collate(batch):
    batch_dataset = [item['dataset'] for item in batch]
    batch_id = [item['id'] for item in batch]
    batch_embedding = torch.stack([item['embedding'] for item in batch])
    return {'dataset': batch_dataset, 'id': batch_id, 'embedding': batch_embedding}

class CSVDataset(Dataset):
    def __init__(self, root_dir, num_workers=1):
        self.data = pd.DataFrame([])
        files_to_process = []
        datasets = []
        for d in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, d)
            files_to_process.append(folder_path)
        for file_name in files_to_process:
            embeddings = torch.load(file_name).to(torch.float32)
            ids = [i for i in range(embeddings.size(0))]
            embedding = [embeddings[i] for i in ids]
            dataset = file_name.replace('_influence_score.pt','')
            df = pd.DataFrame({
                'dataset': [dataset] * embeddings.size(0),
                'id': ids,  # 行号
                'embedding': embedding})
            self.data = pd.concat([self.data, df], ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'dataset': self.data.iloc[idx]['dataset'],
                'id': self.data.iloc[idx]['id'],
                'embedding': self.data.iloc[idx]['embedding']}

root_dir = 'selected_full_data_102030'
dataset = CSVDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1024, collate_fn=custom_collate)

all_embeddings = []
all_datasets = []
all_ids = []
for batch in dataloader:
    embeddings = batch['embedding'].numpy()
    datasets = batch['dataset']
    ids = batch['id']
    all_embeddings.append(embeddings)
    all_datasets.extend(datasets)
    all_ids.extend(ids)

all_embeddings = np.vstack(all_embeddings)
n_clusters = 100  # Adjust this to the number of clusters you want
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(all_embeddings)
labels = kmeans.labels_

distances = kmeans.transform(all_embeddings)
assigned_distances = distances[np.arange(len(all_embeddings)), labels]

results_df = pd.DataFrame({
    'dataset': all_datasets,
    'id': all_ids,
    'label': labels,
    'assigned_distance': assigned_distances
})

output_file = 'out/cluster_LESS_102030_100.pkl'
results_df.to_pickle(output_file)

centers_df = pd.DataFrame(kmeans.cluster_centers_)
centers_output_file = 'out/centers_LESS_102030_100.pkl'
centers_df.to_pickle(centers_output_file)

print(f"Results have been saved to {output_file}")
#import pdb;pdb.set_trace()