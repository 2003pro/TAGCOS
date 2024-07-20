import numpy as np

np.seterr(all='raise')
from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls
import torch

import pandas as pd
import argparse

# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    ''' 
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:

                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    if argmin.item() == A_i.shape[0]:
                        break
                    # print(argmin.item(),A_i.shape[0],index.item())
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        if argmin.item() == A_i.shape[0]:
            break
        # print(b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape,\
        #  torch.matmul(torch.transpose(A_i, 0, 1), x_i).shape)
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
        # print("REsID",resid.shape)

    x_i = x_i.view(-1)
    # print(x_i.shape)
    # print(len(indices))
    for i, index in enumerate(indices):
        # print(i,index,end="\t")
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    # print(x[indices])
    return x

pt_postfix = '_influence_score.pt'

def load_embedding_batch(dataset, indices):
    pt_file = dataset + pt_postfix
    # Assuming each file has embeddings as a dictionary keyed by 'id'
    # Memory-map the tensor
    all_embeddings = torch.load(pt_file)
    # Fetch embeddings by indices
    embeddings_batch = all_embeddings[indices]
    return embeddings_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process file paths for embeddings and centroids.')
    parser.add_argument('file_path', type=str, help='Path to the embeddings data file.')
    parser.add_argument('centroids_path', type=str, help='Path to the centroids data file.')
    parser.add_argument('save_path', type=str, help='Path to the centroids data file.')
    args = parser.parse_args()

    df = pd.read_pickle(args.file_path)
    unique_labels = df['label'].unique()
    final_selected_df = pd.DataFrame()
    centroids = pd.read_pickle(args.centroids_path)
    centroids = torch.tensor(centroids.values, dtype=torch.float32, device="cpu")
    for idx,label in enumerate(unique_labels):
        filtered_df = df[df['label'] == label]
        grouped = filtered_df.groupby('dataset')['id'].apply(list)
        all_embeddings = []
        all_indices = []

        for dataset, indices in grouped.items():
            embeddings = load_embedding_batch(dataset, indices)
            print(f'Loaded embeddings size for label {label} from {dataset}: {embeddings.size()}')
            all_embeddings.append(embeddings)
            all_indices.extend([(dataset, idx) for idx in indices])

        concatenated_embeddings = torch.cat(all_embeddings, dim=0)
        concatenated_embeddings = concatenated_embeddings.to(dtype=torch.float32, device="cpu")
        print(f'Concatenated embeddings size for label {label}: {concatenated_embeddings.size()}')      
        original_indices = np.array(all_indices)
        budget = int(int(concatenated_embeddings.size(0))*0.05)
        #import pdb;pdb.set_trace()
        while budget > 0:
            selected_data = OrthogonalMP_REG_Parallel(torch.transpose(concatenated_embeddings, 0, 1), centroids[label], tol=1e-4, positive=True, nnz=budget, lam=1, device="cpu")
            selected_mask = selected_data.abs() > 0
            selected_count = selected_mask.sum().item()
            budget -= selected_count
            
            # Update selected indices and embeddings
            selected_indices = original_indices[selected_mask.numpy()]
            concatenated_embeddings = concatenated_embeddings[~selected_mask]
            original_indices = original_indices[~selected_mask]

            # Append selected indices to the final DataFrame
            for ds, idx in selected_indices:
                selected_df = filtered_df[(filtered_df['dataset'] == ds) & (filtered_df['id'] == int(idx))]
                final_selected_df = pd.concat([final_selected_df, selected_df])
    final_selected_df.to_pickle(args.save_path)
    
    