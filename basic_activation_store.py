from networkx import selfloop_edges
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import Dataset, load_dataset
import tqdm
import pdb

class BasicActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.cfg = cfg
        

        self.dataset = load_dataset(cfg["dataset_path"], split="train")
        self.dataset = self.dataset.filter(lambda example: example["label"] == 0 or example['label'] == 2)
        self.dataset = self.dataset.shuffle(seed=cfg["seed"])  
        self.dataset_iter = iter(self.dataset)

        self.num_samples_in_batch = cfg['num_samples_in_batch']
        self.num_batches_in_dataset = len(self.dataset) // cfg["num_samples_in_batch"]
        
        self.hook_point = cfg["hook_point"]
        self.layer = cfg["layer"]
        self.device = cfg["device"]
        
        self._tokens_column = self._get_tokens_column()
        
        self.current_batch = 0


    def _get_tokens_column(self):
        col_names = self.dataset.column_names
        if "tokens" in col_names:
            return "tokens"
        elif "input_ids" in col_names:
            return "input_ids"
        elif "text" in col_names:
            return "text"
        elif "content" in col_names:
            return "content"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', 'text' or 'content' column.")
        

    """
    Output shape is (num_samples_in_batch, max_seq_len_in_batch)
    """
    def get_batch_tokens_and_labels(self):
        texts = []
        labels = []
        
        while len(texts) < self.num_samples_in_batch:
            sample = next(self.dataset_iter)
            text = sample[self._tokens_column]
            texts.append(text)
            label = sample["label"] if sample["label"] == 0 else 1
            labels.append(label)
        
        self.current_batch += 1
        batch_tokens = self.model.to_tokens(texts, truncate=True, move_to_device=True, padding_side='left', prepend_bos=False)
        labels = torch.tensor(labels, device=self.device)
        return batch_tokens, labels

    """
    Output shape is (num_samples_in_batch, max_seq_len_in_batch, act_size)
    """
    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.layer + 1,
            )
        return cache[self.hook_point]

    """
    Aggregating along the sequence dimension
    Output shape is (num_samples_in_batch, act_size)
    """
    def _aggregate_activations(self, activations: torch.Tensor):
        if self.cfg["aggregate_function"] == "mean":
            return activations.mean(dim=-2)
        elif self.cfg["aggregate_function"] == "max":
            return activations.max(dim=-2).values
        else:
            raise ValueError(f"Unknown aggregate function: {self.cfg['aggregate_function']}")




    def next_batch(self):
        batch_tokens, labels = self.get_batch_tokens_and_labels()
        
        activations = self.get_activations(batch_tokens)
        
        aggregate_activations = self._aggregate_activations(activations)
        
        # Reshape activations to (num_samples_in_batch * max_seq_len_in_batch, act_size)
        # activations = activations.reshape(-1, self.cfg["act_size"])
        
        
        return  activations, aggregate_activations, labels
    
    def has_next(self):
        return self.current_batch < self.num_batches_in_dataset
    

    def get_testset_activations(self, baseline: bool = False):
        pass