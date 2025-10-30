import sys
from FrequentDirections import FrequentDirections
import torch

from utils import get_dataset_metadata

device = "cuda"
def main(dataset, tau=5):
    num_classes, _, _  = get_dataset_metadata(dataset)
    for i in range(num_classes):
        # load acts
        path = f"checkpoints/gemma-2-2b_layer_24/{dataset}/class_{i}/activations.pt"
        acts = torch.load(path)
        acts = acts.to("cuda")
        acts = acts.reshape(-1, acts.size(-1))

        # fit FD
        fd = FrequentDirections(tau=tau)
        B = fd.fit(acts)
        B = torch.nn.functional.normalize(B, dim=1).cpu()
        path = f"checkpoints/gemma-2-2b_layer_24/{dataset}/class_{i}/frequent_directions_{tau}.pt"
        torch.save(B, path)
        print(f"Finished saved FD to {path}", file=sys.stderr)

        # load dense embeddings
        path = f"checkpoints/dense_representation/{dataset}/class_{i}/all_representations.pt"
        embds = torch.load(path)
        embds = embds.to("cuda")

        # fit FD
        fd = FrequentDirections(tau=tau)
        B = fd.fit(embds)
        B = torch.nn.functional.normalize(B, dim=1).cpu()
        path = f"checkpoints/dense_representation/{dataset}/class_{i}/frequent_directions_{tau}.pt"
        torch.save(B, path)
        print(f"Finished saved FD to {path}", file=sys.stderr)

