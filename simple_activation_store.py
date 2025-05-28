import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import Dataset, load_dataset


class SimpleActivationStore:
    def __init__(
            self,
            model: HookedRootModule,
            cfg: dict,
    ):
        self.model = model
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["context_size"], model.cfg.n_ctx)
        self.batch_size = cfg["train_batch_size"]
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
