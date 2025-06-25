import torch
from torch.utils.data import DataLoader, Dataset
from transformer_lens.hook_points import HookedRootModule
from datasets import load_dataset

class SupervisedActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.cfg = cfg
        

        self.dataset = load_dataset(cfg["dataset_path"], split="train")
        self.testset = load_dataset(cfg["dataset_path"], split="test")
        
        if cfg["filter_labels"] is not None:
            self.filter_labels()
        
        # Shuffle to mix labels
        self.dataset = self.dataset.shuffle(seed=cfg["seed"])
        self.testset = self.testset.shuffle(seed=cfg["seed"])  
        
        self.dataset_iter = iter(self.dataset)
        self.testset_iter = iter(self.testset)

        self.num_samples_in_batch = cfg['num_samples_in_batch']
        
        self.num_batches_in_dataset = len(self.dataset) // self.num_samples_in_batch
        self.num_batchs = 0
        
        self.num_batches_in_testset = len(self.testset) // self.num_samples_in_batch
        self.testset_batch_num = 0
        
        self.hook_point = cfg["hook_point"]
        self.device = cfg["device"]
        
        self._tokens_column = self._get_tokens_column()
        

    def filter_labels(self):
        def filter_func(sample):
            return sample["label"] in self.cfg["filter_labels"]
        self.dataset = self.dataset.filter(filter_func)
        self.testset = self.testset.filter(filter_func)

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
        
        self.num_batchs += 1
        batch_tokens = self.model.to_tokens(texts, truncate=True, move_to_device=True, padding_side='left', prepend_bos=False)
        B, S = batch_tokens.shape
        batch_tokens = batch_tokens[:, :S//2]
        if self.cfg['reactivation']:
            batch_tokens = torch.cat([batch_tokens, batch_tokens], dim=-1)
        batch_labels = torch.tensor(labels, device=self.device, dtype=torch.long)
        return batch_tokens, batch_labels

    """
    Output shape is (num_samples_in_batch, max_seq_len_in_batch, act_size)
    """
    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg['hook_point_layer'] + 1,
            )
        return cache[self.hook_point]



    def next_batch(self):
        batch_tokens, batch_labels = self.get_batch_tokens_and_labels()
        activations = self.get_activations(batch_tokens)
        if self.cfg['reactivation']:
            B, S, C = activations.shape
            activations = activations[:, S//2:, :]
        return activations, batch_labels, batch_tokens
    
    def has_next(self):
        return self.num_batchs < self.num_batches_in_dataset
    

    def next_testset_batch(self):
        texts = []
        labels = []
        
        while len(texts) < self.cfg['num_samples_in_batch']:
            sample = next(self.testset_iter)
            text = sample[self._tokens_column]
            texts.append(text)
            label = sample["label"] if sample["label"] == 0 else 1
            labels.append(label)
        
        batch_tokens = self.model.to_tokens(texts, truncate=True, move_to_device=True, padding_side='left', prepend_bos=False)
        B, S = batch_tokens.shape
        batch_tokens = batch_tokens[:, :S//2]
        if self.cfg['reactivation']:
            batch_tokens = torch.cat([batch_tokens, batch_tokens], dim=-1)
        batch_labels = torch.tensor(labels, device=self.device, dtype=torch.long)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg['hook_point_layer'] + 1,
            )
        activations = cache[self.hook_point]
        if self.cfg['reactivation']:
            B, S, C = activations.shape
            activations = activations[:, S//2:, :]

        self.testset_batch_num += 1

        return activations, batch_labels, batch_tokens

    def has_next_testset(self):
        return self.testset_batch_num < self.num_batches_in_testset

    def reset_testset_iter(self):
        self.testset_iter = iter(self.testset)
        self.testset_batch_num = 0
    
class UnsupervisedActivationStore:
    def __init__(
            self,
            model: HookedRootModule,
            cfg: dict,
    ):
        self.model = model
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["context_size"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.dtype = cfg["dtype"]
        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")
        
    def get_batch_tokens(self):
        all_tokens = []
        while len(all_tokens) < self.batch_size * self.context_size:
            sample = next(self.dataset)
            if self.tokens_column == "text":
                # Removed BOS token as GPT2 wasn't trained with it
                # See reference: https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html#transformer_lens.HookedTransformer.HookedTransformer.to_tokens
                tokens = self.model.to_tokens(sample[self.tokens_column], truncate=True, move_to_device=True, prepend_bos=False).squeeze(0)
            else:
                tokens = sample[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.batch_size * self.context_size]
        return token_tensor.view(self.batch_size, self.context_size)
    
    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["hook_point_layer"] + 1,
            )
        return cache[self.hook_point]
    
    def next_batch(self):
        batch_tokens = self.get_batch_tokens()
        batch_acts = self.get_activations(batch_tokens)
        return batch_tokens, batch_acts
    
class dbpedia_14_dataset(Dataset):
    def __init__(self, cfg, model):
        self.cfg = cfg