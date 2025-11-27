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
              "async", "await", "try", "except", "for", "while", 
              "func", "var", "let", "const", "null", "true", "false", 
              "struct", "enum", "include", "template"]

def create_dataset_tokens_mass(
        model: HookedRootModule,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        tokens_ids: list[int],
        dataset_size: int = int(1e4)
        ):
    """
    Creates tensors X: [dataset_size, hidden_dim] of last activation for every layer and Y: [dataset_size] of -log token mass for given token ids.
    Assumes dataset has "input_ids" key which is tokenized input in chat format, and that input_ids are bounded by some L_max.
    """
    model = model.to("cuda")

    batch_size = 32
    num_iterations = dataset_size // batch_size
    
    L_max = 256
    
    hookpoints = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)] # all layer resuidual stream
    
    hf_model_name = get_official_model_name(model.cfg.model_name)
    dataset = load_from_disk(f"data/preprocessed/{dataset_name.replace('/', '_')}/{hf_model_name.replace('/', '_')}_L={L_max}")
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left" # so the last act corresponds to the last token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token # doen't matter what we pad with since we will mask it out
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest') # collate examples to batch by padding to longest in batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)
    
    mlog_tokens_mass_list = []
    acts_list_per_layer = defaultdict(list)
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
        
        # record last acts
        for hookpoint in hookpoints:
            acts = cache[hookpoint]  # [batch, seq_len, d_model]
            acts_list_per_layer[hookpoint].append(acts[:, -1, :].cpu())  # [batch, d_model]

        # record token mass for target tokens
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        probs = torch.nn.functional.softmax(last_logits, dim=-1)  # [batch, vocab_size]
        tokens_mass = probs[:, tokens_ids].sum(dim=-1) # [batch]
        mlog_tokens_mass = -torch.log(tokens_mass + 1e-10)  # [batch]
        mlog_tokens_mass_list.append(mlog_tokens_mass.cpu())


        print(f"Processed batch {i+1}/{num_iterations}", file=sys.stderr)
    
    Xs = []
    for hookpoint in hookpoints:
        acts_list = acts_list_per_layer[hookpoint]
        X = torch.cat(acts_list, dim=0)  # [dataset_size, hidden_dim]
        Xs.append(X)

    Y = torch.cat(mlog_tokens_mass_list, dim=0)  # [dataset_size]
    
    return Xs, Y



def fit_ridge_regression(X: torch.Tensor, Y: torch.Tensor):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, Y)
    return ridge


def create_dataset_perplexity(
        model: HookedRootModule,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        dataset_size: int = int(1e4)
        ):
    """
    Creates a dataset for every layer of last activations Xs and perplexity Y.
    Assumes dataset has "input_ids" key which is tokenized input in chat format.
    """
    model = model.to("cuda")
    
    batch_size = 32
    num_iterations = dataset_size // batch_size
    
    L_max = 256
    
    hookpoints = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    
    hf_model_name = get_official_model_name(model.cfg.model_name)
    dataset = load_from_disk(f"data/preprocessed/{dataset_name.replace('/', '_')}/{hf_model_name.replace('/', '_')}_L={L_max}")
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left" # so the last act corresponds to the last token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token # doen't matter what we pad with since we will mask it out
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest') # collate examples to batch by padding to longest in batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)
    
    acts_list_per_layer = defaultdict(list)
    perplexities_list = []
    for i in range(num_iterations):
        encoded_batch = next(data_loader_iter)
        tokens = encoded_batch.input_ids.to("cuda")  # [batch, seq_len]
        att_mask = encoded_batch.attention_mask.to("cuda")  # [batch, seq_len]
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, 
                                                 names_filter=hookpoints,
                                                 attention_mask=att_mask)
        
        # record last acts
        for hookpoint in cache.keys():
            acts = cache[hookpoint]  # [batch, seq_len, d_model]
            acts_list_per_layer[hookpoint].append(acts[:, -1, :].cpu())  # [batch, d_model]

        # record perplexity
        target_tokens = tokens.clone()[:, 1:] # [batch, seq_len - 1]
        minus_log_probs = (-torch.nn.functional.log_softmax(logits, dim=-1))[:, :-1, :]  #  [batch, seq_len - 1, vocab_size]
        target_log_probs = torch.gather(minus_log_probs,
                                        dim=-1,
                                        index=target_tokens.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len - 1]
        
        att_mask = att_mask[:, :-1] # adjust attention mask on the tokens we measured log probs for [batch, seq_len - 1]
        masked_target_log_probs = target_log_probs * att_mask # [batch, seq_len - 1], 
        sums = masked_target_log_probs.sum(dim=-1)  # [batch]
        counts  = att_mask.sum(dim=-1).clamp(min=1)  # [batch]
        perplexity = torch.exp(sums / counts)  # [batch]
        perplexities_list.append(perplexity.cpu())
        
        print(f"Processed batch {i+1}/{num_iterations}", file=sys.stderr)
    
    Xs = []
    for hookpoint in hookpoints:
        acts_list = acts_list_per_layer[hookpoint]
        X = torch.cat(acts_list, dim=0).cpu()  # [dataset_size, hidden_dim]
        Xs.append(X)
    Y = torch.cat(perplexities_list, dim=0).cpu()  # [dataset_size]

    return Xs, Y

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

        if task == "pred_tokens_mass":
            cfg = AutoConfig.from_pretrained(get_official_model_name(model_name))
            print("-log mass stats: mean ", torch.mean(Y).item(), " std ", torch.std(Y).item())
            vocab_size = cfg.vocab_size
            mass = torch.exp(-Y)
            scaled_mass = mass * vocab_size
            print("Scaled mass stats: mean ", torch.mean(scaled_mass).item(), " std ", torch.std(scaled_mass).item())
            Y = scaled_mass

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
                            dataset: IterableDataset,
                            dataset_size: int = int(1e4)) -> torch.Tensor:
    model = model.to("cuda")
    batch_size = 32

    # prepare dataset
    dataset = dataset.shuffle(seed=42)
    def keep_only_user_prompt(example: dict) -> dict:
        new_conv = []
        new_conv.append(example["conversation"][0]) 
        return {"conversation": new_conv}
    dataset = dataset.map(keep_only_user_prompt)
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

    if task == "pred_tokens_mass":
        cfg = AutoConfig.from_pretrained(get_official_model_name(llm_model_name))
        vocab_size = cfg.vocab_size
        mass = torch.exp(-Y)
        scaled_mass = mass 
        Y = scaled_mass

    if project_dim is not None:
        embeddings = project(embeddings, project_dim)

    X_train, X_test, Y_train, Y_test = train_test_split(embeddings, Y, test_size=0.2, random_state=42)

    ridge = fit_ridge_regression(X_train, Y_train)

    relative_error = compute_relative_error(ridge, X_test, Y_test)
    r2_score = ridge.score(X_test, Y_test)

    return relative_error, r2_score


def create_dataset_lengths(
        model: HookedRootModule,
        model_name: str,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        dataset_size: int = 1000
        ):
    """
    Creates a dataset for every layer of last activations Xs and perplexity Y.
    Assumes dataset has "input_ids" key which is tokenized input in chat format.
    """
    model = model.to("cuda")
    
    batch_size = 32
    num_iterations = dataset_size // batch_size
    
    L_max = 256
    
    hookpoints = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    
    hf_model_name = get_official_model_name(model_name)
    dataset = load_from_disk(f"data/preprocessed/{dataset_name.replace('/', '_')}/{hf_model_name.replace('/', '_')}_L={L_max}")
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left" # so the last act corresponds to the last token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token # doen't matter what we pad with since we will mask it out
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest') # collate examples to batch by padding to longest in batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)
    
    acts_list_per_layer = defaultdict(list)
    lengths_list = []
    for i in range(num_iterations):
        encoded_batch = next(data_loader_iter)
        tokens = encoded_batch.input_ids.to("cuda")  # [batch, seq_len]
        att_mask = encoded_batch.attention_mask.to("cuda")  # [batch, seq_len]
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, 
                                                 names_filter=hookpoints,
                                                 attention_mask=att_mask)
        
        # record last acts
        for hookpoint in cache.keys():
            acts = cache[hookpoint]  # [batch, seq_len, d_model]
            acts_list_per_layer[hookpoint].append(acts[:, -1, :].cpu())  # [batch, d_model]

        # record response lengths
        query_length = tokens.shape[1] 
        response = model.generate(tokens,
                                  max_new_tokens=(model.cfg.n_ctx - query_length - 500),
                                  verbose=False)
        response_only = response[:, query_length:]  # [batch, gen_seq_len]
        not_eos = (response_only != tokenizer.eos_token_id)
        lengths = not_eos.sum(dim=-1) + 1  # +1 to account for eos token; [batch]
        lengths_list.append(lengths.cpu())
        
        print(f"Processed batch {i+1}/{num_iterations}", file=sys.stderr)
    
    Xs = []
    for hookpoint in hookpoints:
        acts_list = acts_list_per_layer[hookpoint]
        X = torch.cat(acts_list, dim=0).cpu()  # [dataset_size, hidden_dim]
        Xs.append(X)
    Y = torch.cat(lengths_list, dim=0).cpu()  # [dataset_size]

    return Xs, Y
        


 
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