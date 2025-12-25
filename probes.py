from collections import defaultdict
import sys
from transformer_lens.hook_points import HookedRootModule
from transformer_lens.loading_from_pretrained import get_official_model_name
from datasets import IterableDataset, Dataset, load_from_disk
import torch
from torch.nn import Identity
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig, PreTrainedModel


tl_model_names = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen2.5-7B-Instruct",
    "mistral": "mistral-7b-instruct",
    "phi": "phi-3",
}

def get_hf_model_name(tl_model_name: str) -> str:
    return get_official_model_name(tl_model_name)

code_words = ["def", "class", "import", "return", "lambda", 
              "async", "await","func", "var", "let", "const", "null", 
              "struct", "enum", "include", "template", "typename"]

math_words = ["assume", "let", "therefore", "hence", "thus", "suppose",
                "expand", "differentiate", "integrate", "factor", "solve",
                "substitute", "equation", "expression"]

def fit_estimator(X: torch.Tensor, Y: torch.Tensor, task_type="regression"):
    est = None
    if task_type == "regression":
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, Y) 
        est = ridge
    elif task_type == "classification":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, Y)
        est = clf
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return est

def eval(model_name, 
         task, 
         dataset, 
         task_type="regression",
         project_dim=None, 
         pooling_strategy="last"):
    
    # load labels
    Y = torch.load(f"data/pred_gen/{dataset}/{model_name}/Y_{task}.pt") # [dataset_size]
    
    # load acts
    path = f"data/pred_gen/{dataset}/{model_name}"
    layers = [int(folder.split("_")[1]) for folder in os.listdir(path) 
                if os.path.isdir(os.path.join(path, folder))]
    layers = sorted(layers)
    rel_errs = []
    r2_scores = []
    for layer in layers:
        cur_Y = Y.clone()
        X_name = f"X{('_' + pooling_strategy) if pooling_strategy else ''}.pt"
        X = torch.load(f"{path}/layer_{layer}/{X_name}") # [dataset_size, hidden_dim]
        X = X[:cur_Y.shape[0], :] # take only as many examples as in Y
        X, cur_Y, num_valid_examples = filter_valid_examples(X, cur_Y)
        
        if project_dim is not None:
            X = project(X, project_dim)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, cur_Y, test_size=0.2, random_state=42)
        estimator = fit_estimator(X_train, Y_train, task_type)

        Y_pred = estimator.predict(X_test)
        relative_error = compute_relative_error(Y_pred, Y_test) if task_type == "regression" else -1.0
        score = estimator.score(X_test, Y_test)

        print(f"Layer: {layer}, rel_err (reg only): {relative_error:.2f}%, score (R^2 for reg /acc for clf): {score:.4f}")
        rel_errs.append(relative_error)
        r2_scores.append(score)
    
    return layers, rel_errs, r2_scores

def eval_baseline(baseline_model_name: str,
                  dataset_name: str,
                  tl_model_name: str, 
                  task, 
                  project_dim=None,
                  task_type="regression"):
    Y = torch.load(f"data/pred_gen/{dataset_name}/{tl_model_name}/Y_{task}.pt")
    num_examples = Y.shape[0]
    path = f"data/embeddings/{baseline_model_name}/{dataset_name}/{tl_model_name}_L=256/embeddings.pt"
    embeddings = torch.load(path)  # [dataset_size, embedding_dim]
    embeddings = embeddings[:num_examples, :]  # take only as many examples as in Y
    embeddings, Y, num_valid_examples = filter_valid_examples(embeddings, Y)
    print(f"Num valid examples: {num_valid_examples} / {num_examples}", file=sys.stderr)
    
    if project_dim is not None:
        embeddings = project(embeddings, project_dim)
    
    X_train, X_test, Y_train, Y_test = train_test_split(embeddings, Y, test_size=0.2, random_state=42)

    estimator = fit_estimator(X_train, Y_train, task_type)

    Y_pred = estimator.predict(X_test)
    relative_error = compute_relative_error(Y_pred, Y_test) if task_type == "regression" else -1.0
    score = estimator.score(X_test, Y_test)
    print(f"Baseline Model: {baseline_model_name}, rel_err: {relative_error:.2f}%, R^2: {score:.4f}")
    return relative_error, score

def compute_relative_error(Y_pred: torch.Tensor, Y_test: torch.Tensor) -> float:
    relative_error = torch.mean(torch.abs(Y_test - Y_pred) / (Y_test + 1e-10)).item() * 100
    return relative_error

def create_dataset_baseline(model: SentenceTransformer,
                            dataset_name: str,
                            tl_model_name: str,
                            dataset_size: int = int(1e4),
                            L_max: int = 256,
                            batch_size: int = 32) -> torch.Tensor:
    model = model.to("cuda")
    
    dataset = load_from_disk(f"data/preprocessed/{dataset_name}/{tl_model_name}_L={L_max}")
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
      
def create_datasets(
        model: PreTrainedModel,
        tl_model_name: str,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        pooling_strategies: list[str] = ["last"],
        tokens_ids: list[int] = None,
        dataset_size: int = int(1e4),
        record_nll_telemetry: bool = True,
        record_tokens_mass_telemetry: bool = True,
        record_length_telemetry: bool = True,
        batch_size: int = 32,
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
    
    # workaround for HF Transformers output_hidden_states=True outputing the final layer output after the final LN
    identity = Identity()
    model.model.norm = identity
    
    # load dataset
    dataset = load_from_disk(f"data/preprocessed/{dataset_name}/{tl_model_name}_L={L_max}")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["input_ids"]]) # keep only input_ids for batching
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left" # so the last act corresponds to the last token
    tokenizer.pad_token = tokenizer.eos_token 
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest') 
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)

    dataset_size = min(len(dataset), dataset_size)
    num_iterations = dataset_size // batch_size
    
    Ys = defaultdict(list)
    acts_per_layer_per_pooling = defaultdict(lambda: defaultdict(list))  # layer_idx -> pooling_strategy -> list of tensors
    for i in range(num_iterations):
        inputs = next(data_loader_iter)
        inputs = {key: val.to("cuda") for key, val in inputs.items()}
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states[1:]  # exclude embedding layer, list of [batch, seq_len, d_model]

        # record acts based on pooling strategy
        for layer_idx in range(model.config.num_hidden_layers):
            acts = hidden_states[layer_idx] # [batch, seq_len, d_model]
            for pooling_strategy in pooling_strategies:
                acts_to_save = pool_activations(acts, inputs["attention_mask"], pooling_strategy)  # [batch, d_model]
                acts_per_layer_per_pooling[layer_idx][pooling_strategy].append(acts_to_save.cpu())  # [batch, d_model]

        # # generate responses
        # responses = generate(**inputs, model=model, eos_token_id=tokenizer.eos_token_id)  # [batch, gen_len]
        # Ys["response_token_ids"].append(responses.cpu()) 
        
        # # record length 
        # lengths = compute_response_length(responses, tokenizer.eos_token_id)  # [batch]
        # Ys["lengths"].append(lengths.cpu())

        print(f"Processed batch {i+1}/{num_iterations}", file=sys.stderr)
    
    for layer_idx, pooling_dict in acts_per_layer_per_pooling.items():
        for pooling_strategy, acts_list in pooling_dict.items():
            acts_tensor = torch.cat(acts_list, dim=0)  # [dataset_size, d_model]
            acts_per_layer_per_pooling[layer_idx][pooling_strategy] = acts_tensor  # [dataset_size, d_model]
    Ys_cat = {}
    for label_name, lst_tensors in Ys.items():
        if label_name == "response_token_ids":
            Ys_cat[label_name] = concat_tensors_of_different_lengths(lst_tensors, padding_value=tokenizer.eos_token_id) # [dataset_size, max_gen_len]
        else:
            Ys_cat[label_name] = torch.cat(lst_tensors, dim=0)  # [dataset_size] 
    
    return acts_per_layer_per_pooling, Ys_cat

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

def generate(input_ids: torch.Tensor, 
             attention_mask: torch.Tensor, 
             model: PreTrainedModel, 
             eos_token_id: int,
             max_new_tokens: int = 2048) -> torch.Tensor:
    
    query_length = input_ids.shape[1]
    response = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens, # max generation length
                                do_sample=False, # greedy decoding for reproducibility
                                pad_token_id=eos_token_id # so we can know if a generation was truncated
                            ) # [batch, query_length + gen_len]
    response_without_query = response[:, query_length:]  # [batch, gen_len]
    return response_without_query

def compute_response_length(responses: torch.Tensor,  eos_token_id: int) -> torch.Tensor:
    # responses: [batch, gen_len]
    not_eos = (responses != eos_token_id)
    lengths = not_eos.sum(dim=-1) + 1  # +1 to account for last eos token; [batch]
    is_truncated = (responses[:, -1] != eos_token_id) # [batch]
    lengths[is_truncated] = -1  # mark truncated generations with -1
    return lengths

def concat_tensors_of_different_lengths(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    num_rows = sum(tensor.size(0) for tensor in tensors)
    max_cols = max(tensor.size(1) for tensor in tensors)
    dtype = tensors[0].dtype
    device = tensors[0].device
    result = torch.full((num_rows, max_cols), padding_value, dtype=dtype, device=device)
    offset = 0
    for tensor in tensors:
        rows, cols = tensor.size()
        result[offset:offset + rows, :cols] = tensor
        offset += rows
    return result

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

def filter_valid_examples(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filters out examples where Y is -1.
    X: [num_samples, dim]
    Y: [num_samples]
    Returns: X_filtered: [num_valid_samples, dim], Y_filtered: [num_valid_samples]
    """
    valid_mask = (Y != -1)
    X_filtered = X[valid_mask]
    Y_filtered = Y[valid_mask]
    num_valid_examples = X_filtered.shape[0]
    return X_filtered, Y_filtered, num_valid_examples