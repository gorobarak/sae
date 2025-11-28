from collections import defaultdict
import sys
from transformer_lens.hook_points import HookedRootModule
from transformer_lens.loading_from_pretrained import get_official_model_name
from datasets import IterableDataset, Dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig



code_words = ["def", "class", "import", "return", "lambda", 
              "async", "await","func", "var", "let", "const", "null", 
              "struct", "enum", "include", "template", "typename"]

def fit_ridge_regression(X: torch.Tensor, Y: torch.Tensor):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, Y)
    return ridge

def eval(model_name, task, project_dim=None):
    path = f"data/{task}/{model_name}"
    hookpoints = os.listdir(path)
    hookpoints = sorted(hookpoints, key=lambda x: int(x.split(".")[1]))
    layers = [int(hookpoint.split(".")[1]) for hookpoint in hookpoints]
    rel_errs = []
    r2_scores = []
    for hookpoint in hookpoints:
        X = torch.load(f"{path}/{hookpoint}/X.pt") # [dataset_size, hidden_dim]
        Y = torch.load(f"{path}/{hookpoint}/Y.pt") # [dataset_size]
        
        if task == "pred_perplexity":
            Y = torch.log(Y + 1e-10)  # predict log perplexity to stabilize training

        if project_dim is not None:
            X = project(X, project_dim)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        ridge = fit_ridge_regression(X_train, Y_train)

        relative_error = compute_relative_error(ridge, X_test, Y_test)
        r2_score = ridge.score(X_test, Y_test)

        print(f"Model: {model_name}, Hookpoint: {hookpoint}, Test relative error: {relative_error:.4f}, R^2: {r2_score:.4f}")
        rel_errs.append(relative_error)
        r2_scores.append(r2_score)

    return layers, rel_errs, r2_scores

def compute_relative_error(ridge: Ridge, X_test: torch.Tensor, Y_test: torch.Tensor) -> float:
    Y_pred = ridge.predict(X_test)
    relative_error = torch.mean(torch.abs(Y_test - Y_pred) / (Y_test + 1e-10)).item()
    return relative_error

def create_dataset_baseline(model: SentenceTransformer,
                            dataset_name: str,
                            llm_hf_model_name: str,
                            dataset_size: int = int(1e4),
                            L_max=256) -> torch.Tensor:
    model = model.to("cuda")
    batch_size = 32
    
    dataset = load_from_disk(f"data/preprocessed/{dataset_name.replace('/', '_')}/{llm_hf_model_name.replace('/', '_')}_L={L_max}")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.iter(batch_size=batch_size)
    embeddings_list = []
    
    for i in range(dataset_size // batch_size):
        
        conversations = (next(dataset))["conversation"]
        texts = convert_conversations_to_texts(conversations)

        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device="cuda",
            show_progress_bar=False
        )  # [batch, embedding_dim]
                
        embeddings_list.append(embeddings.cpu())

        print(f"Processed embeddings batch {i+1}/{dataset_size // batch_size}", file=sys.stderr)
    
    all_embds = torch.cat(embeddings_list, dim=0)  # [dataset_size, embedding_dim]
    return all_embds

def convert_conversations_to_texts(conversations: list[list[dict]]) -> list[str]:
    texts = []
    for conv in conversations:
        text = ''
        for msg in conv:
            text += msg['content'] + "\n\n"
        texts.append(text)
    return texts

def eval_baseline(basline_model, llm_model_name: str , task, project_dim=None):

    path = f"data/{basline_model}/allenai/WildChat-1M/embeddings.pt"
    embeddings = torch.load(path)  # [dataset_size, embedding_dim]
    Y = torch.load(f"data/{task}/{llm_model_name}/blocks.0.hook_resid_post/Y.pt")

    if task == "pred_perplexity":
        Y = torch.log(Y + 1e-10)  # predict log perplexity to stabilize training

    if project_dim is not None:
        embeddings = project(embeddings, project_dim)

    X_train, X_test, Y_train, Y_test = train_test_split(embeddings, Y, test_size=0.2, random_state=42)

    ridge = fit_ridge_regression(X_train, Y_train)

    relative_error = compute_relative_error(ridge, X_test, Y_test)
    r2_score = ridge.score(X_test, Y_test)

    return relative_error, r2_score
      
def create_datasets(
        model: HookedRootModule,
        hf_model_name: str,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        pooling_strategy: str = "last",
        tokens_ids: list[int] = None,
        dataset_size: int = int(1e4),
        record_nll_telemetry: bool = True,
        record_tokens_mass_telemetry: bool = True,
        record_length_telemetry: bool = True,
        batch_size: int = 16,
        L_max: int = 256
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Creates tensor X of residual stream activation for every layer using the pooling strategy pooling_strategy 
    Creates dict of tensors Y with keys the different tasks
        - "pred_perplexity": Y["pred_perplexity"] is perplexity tensor
        - "pred_tokens_mass": Y["pred_tokens_mass"] is -log token mass tensor for given token ids
        - "pred_lengths": Y["pred_lengths"] is response lengths tensor

    Assumes dataset was preprocessed and has "input_ids" key which is tokenized input in chat format, and input_ids is bounded by some L_max.
    Preprocessd dataset should be in data/preprocessed/{dataset_name}/{model_name}_L={L_max}
    """

    model = model.to("cuda")

    num_iterations = dataset_size // batch_size
    
    
    hookpoints = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)] # all layer resuidual stream
    
    # load dataset
    dataset = load_from_disk(f"data/preprocessed/{dataset_name.replace('/', '_')}/{hf_model_name.replace('/', '_')}_L={L_max}")
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left" # so the last act corresponds to the last token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token # doen't matter what we pad with since we will mask it out
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest') # collate examples to batch by padding to longest in batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)
    
    Ys = defaultdict(list)
    acts_per_layer = defaultdict(list)
    for i in range(num_iterations):
        encoded_batch = next(data_loader_iter)
        tokens = encoded_batch.input_ids.to("cuda")  # [batch, seq_len]
        att_mask = encoded_batch.attention_mask.to("cuda")  # [batch, seq_len]
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=hookpoints,
                attention_mask=att_mask
            )
        
        # record acts based on pooling strategy
        for hookpoint in hookpoints:
            acts = cache[hookpoint] # [batch, seq_len, d_model]
            acts_to_save = pool_activations(acts, att_mask, pooling_strategy)  # [batch, d_model]
            acts_per_layer[hookpoint].append(acts_to_save.cpu())  # [batch, d_model]

        # record labels for different tasks
        if record_nll_telemetry:
            Y_mnll = record_nll(tokens, att_mask, logits)  # [batch]
            Ys["pred_nll"].append(Y_mnll.cpu())
        if record_tokens_mass_telemetry:
            Y_tokens_mass = record_tokens_mass(logits, tokens_ids)  # [batch]
            Ys["pred_tokens_mass"].append(Y_tokens_mass.cpu())
        if record_length_telemetry:
            Y_lengths = record_length(tokens, model, tokenizer)  # [batch]
            Ys["pred_length"].append(Y_lengths.cpu())

        print(f"Processed batch {i+1}/{num_iterations}", file=sys.stderr)
    
    Xs_cat = {key: torch.cat(acts_per_layer[key], dim=0) for key in acts_per_layer.keys()} # [dataset_size, hidden_dim]
    Ys_cat = {key: torch.cat(Ys[key], dim=0) for key in Ys.keys()} # [dataset_size]
    
    return Xs_cat, Ys_cat

def pool_activations(acts: torch.Tensor, att_mask: torch.Tensor, pooling_strategy: str) -> torch.Tensor:
    """
    Pools activations based on pooling_strategy.
    acts: [batch, seq_len, d_model]
    att_mask: [batch, seq_len]
    pooling_strategy: "last", "mean", "max"
    Returns: pooled_acts: [batch, d_model]
    """
    if pooling_strategy == "last":
        # because we pad on the left, the last token is at the last position
        pooled_acts = acts[:, -1, :]  # [batch, d_model]
    elif pooling_strategy == "mean":
        masked_acts = acts * att_mask.unsqueeze(-1) # [batch, seq_len, d_model]
        sums = masked_acts.sum(dim=1) # sum over sequence dim [batch, d_model]
        counts = att_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        pooled_acts = sums / counts  # [batch, d_model]
    elif pooling_strategy == "max":
        mask = att_mask.unsqueeze(-1).bool()  # [batch, seq_len, 1]
        acts_masked = acts.float().masked_fill(~mask, float("-inf"))  # [batch, seq_len, d_model]
        pooled_acts = torch.max(acts_masked, dim=1).values  # max over sequence dim [batch, d_model]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return pooled_acts

def record_tokens_mass(logits: torch.Tensor, tokens_ids: list[int]) -> torch.Tensor:
    last_logits = logits[:, -1, :]  # [batch, vocab_size]
    probs = torch.nn.functional.softmax(last_logits, dim=-1)  # [batch, vocab_size]
    tokens_mass = probs[:, tokens_ids].sum(dim=-1) # [batch]
    mlog_mass = -torch.log(tokens_mass + 1e-10)  # [batch]
    return mlog_mass

def record_nll(tokens: torch.Tensor, att_mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    # tragets are all tokens except first
    target_tokens = tokens[:, 1:] # [batch, seq_len - 1]
    
    # we take log prob on all tokens except last
    minus_log_probs = (-torch.nn.functional.log_softmax(logits, dim=-1))[:, :-1, :]  #  [batch, seq_len - 1, vocab_size] 
    target_log_probs = torch.gather(minus_log_probs,
                                    dim=-1,
                                    index=target_tokens.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len - 1]
    
    att_mask = att_mask[:, :-1] # adjust attention mask on the tokens we measured log probs for [batch, seq_len - 1]
    masked_target_log_probs = target_log_probs * att_mask # [batch, seq_len - 1], 
    sums = masked_target_log_probs.sum(dim=-1)  # [batch]
    counts  = att_mask.sum(dim=-1).clamp(min=1)  # [batch]
    mean_nll = sums / counts  # [batch]
    return mean_nll

def record_length(tokens: torch.Tensor, model: HookedRootModule, tokenizer: AutoTokenizer) -> torch.Tensor:
    query_length = tokens.shape[1] 
    response = model.generate(tokens,
                                max_new_tokens=(model.cfg.n_ctx - query_length - 500),
                                verbose=False)
    response_only = response[:, query_length:]  # [batch, gen_seq_len]
    not_eos = (response_only != tokenizer.eos_token_id)
    lengths = not_eos.sum(dim=-1) + 1  # +1 to account for eos token; [batch]
    return lengths

def project(X: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Projects X to target_dim using radnom gaussian matrix.
    X: [num_samples, original_dim]
    Returns: X_projected: [num_samples, target_dim]
    """
    original_dim = X.shape[1]
    torch.manual_seed(42) # for reproducibility
    projection_matrix = (target_dim ** -0.5) * torch.randn(original_dim, target_dim)

    X_projected = X @ projection_matrix  # [num_samples, target_dim]
    return X_projected