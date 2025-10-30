import torch
import numpy as np
from utils import get_dataset_metadata, class_idx_to_class_name

class FrequentDirections():
    def __init__(self, tau):
        self.tau = tau # how many directions to keep
        self.m = tau + 1 # sketch size
        self.B = None  # sketch matrix


    def fit(self, A):
        """
        Fit the Frequent Directions sketch to the data matrix A.
        
        Args:
            A (torch.Tensor): Input data matrix of shape (n, d).
        Returns:
            torch.Tensor: The frequent directions of shape (tau, d)
        """
        n, d = A.shape
        self.B = torch.zeros((self.m, d), device=A.device)  # Initialize B with zeros
        for i in range(n):
            self.B[-1] = A[i]  # Insert new row at the bottom of B
            # Perform SVD on B
            U, S, Vt = torch.linalg.svd(self.B, full_matrices=False)
            # Shrink first tau singular values. zeroing all values past it.
            # example: if tau=2 and S=[5,4,3,2,1], then we want to shrink by 3, resulting in [sqrt(16), sqrt(7), 0, 0, 0]
            tau_singular_value = S[self.tau]
            shrunk_singular_values = torch.sqrt(torch.clamp(S**2 - tau_singular_value**2, min=0))

            # Reconstruct B
            self.B = torch.diag(shrunk_singular_values) @ Vt

        # Return the frequent directions
        return self.B[:self.tau]


def analyze_fds(dataset, glove50emb, glove50embwords, model, tau=10):
    num_classes, _, _ = get_dataset_metadata(dataset)
    lines_acts = [f"Frequent Directions analysis for {dataset} classes, tau={tau}\n"]
    lines_embds = [f"Frequent Directions analysis for {dataset} classes, tau={tau}\n"]
    for class_idx in range(num_classes):
        
        # analyze fd of activations
        fd_acts = torch.load(f"checkpoints/gemma-2-2b_layer_24/{dataset}/class_{class_idx}/frequent_directions_{tau}.pt")
        all_logits = model.unembed(model.ln_final(fd_acts))
        topk = torch.topk(all_logits, k=3, dim=-1)
        lines_acts.append(f"Class \"{class_idx_to_class_name(class_idx, dataset)}\" top projected tokens per direction:")
        for i, (token_ids, logits) in enumerate(zip(topk.indices, topk.values)):
            lines_acts.append(f"  Direction {i + 1}:")
            for token_id, logit in zip(token_ids, logits):
                token = repr(model.tokenizer.convert_ids_to_tokens(int(token_id)))
                lines_acts.append(f"    {token}: {logit:.2f}")

        # analyze fd of embeddings
        fd_embds = torch.load(f"checkpoints/dense_representation/{dataset}/class_{class_idx}/frequent_directions_{tau}.pt")
        scores = fd_embds @ glove50emb
        topk = torch.topk(scores, k=3, dim=-1)
        lines_embds.append(f"Class \"{class_idx_to_class_name(class_idx, dataset)}\" top similar GloVe tokens per direction:")
        for i, (token_ids, logits) in enumerate(zip(topk.indices, topk.values)):
            lines_embds.append(f"  Direction {i + 1}:")
            for token_id, logit in zip(token_ids, logits):
                token = glove50embwords[int(token_id)]
                lines_embds.append(f"    {token}: {logit:.2f}")
        
        lines_acts.append("")
        lines_embds.append("")

    with open(f"checkpoints/gemma-2-2b_layer_24/{dataset}/fd_analysis_{tau}.txt", "w") as f:
        f.write("\n".join(lines_acts))
    with open(f"checkpoints/dense_representation/{dataset}/fd_analysis_{tau}.txt", "w") as f:
        f.write("\n".join(lines_embds))
    print(f"Saved FD analysis to checkpoints/gemma-2-2b_layer_24/{dataset}/fd_analysis_{tau}.txt")
    print(f"Saved FD analysis to checkpoints/dense_representation/{dataset}/fd_analysis_{tau}.txt")