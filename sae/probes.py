from collections import defaultdict
from datetime import datetime
import sys
from datasets import load_from_disk
import torch
from torch.nn import Identity
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
import os
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
)
from tqdm import tqdm


def fit_estimator(X: torch.Tensor, Y: torch.Tensor, task_type="regression"):
    est = None
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    if task_type == "regression":
        ridge = (
            Ridge()
        )  # default hyperparameters specifically L2 regularization coefficient = 1.0
        ridge.fit(X, Y)
        est = ridge
    elif task_type == "classification":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, Y)
        est = clf
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return est


def eval(
    model_name,
    task,
    dataset,
    task_type="regression",
    project_dim=None,
    pooling_strategy="last",
):
    BASE_PATH = "/home/yandex/APDL2425a/group_12/gorodissky/sae/data"
    path = f"{BASE_PATH}/{task}/{dataset}/{model_name}"
    cpts = os.listdir(path)
    cpts = sorted(
        cpts, key=lambda version: datetime.strptime(version, "%Y_%m_%d-%H:%M")
    )
    latest_cpt = cpts[-1]
    path = os.path.join(path, latest_cpt)

    Y = torch.load(f"{path}/Y_{task}.pt").to(torch.float32)  # [dataset_size]
    num_examples = Y.shape[0]
    valid_mask = Y != -1
    print("Number of valid examples:", valid_mask.sum().item(), " / ", num_examples)
    Y = Y[valid_mask]

    cfg = AutoConfig.from_pretrained(model_name)
    num_layers = cfg.num_hidden_layers
    rel_errs = []
    r2_scores = []
    
    for layer in range(num_layers):

        X = torch.load(f"{path}/X_{pooling_strategy}_layer={layer}.pt").to(torch.float32)  # [dataset_size, hidden_dim]
        X = X[valid_mask]

        if project_dim is not None:
            X = project(X, project_dim)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        estimator = fit_estimator(X_train, Y_train, task_type)

        Y_pred = estimator.predict(X_test)
        relative_error = (
            compute_relative_error(Y_pred, Y_test)
            if task_type == "regression"
            else -1.0
        )
        score = estimator.score(X_test, Y_test)

        print(
            f"Layer: {layer}, rel_err (reg only): {relative_error:.2f}%, score (R^2 for reg /acc for clf): {score:.2f}"
        )
        rel_errs.append(relative_error)
        r2_scores.append(score)

    return list(range(num_layers)), rel_errs, r2_scores


def eval_baseline(
    baseline_model_name: str,
    dataset_name: str,
    tl_model_name: str,
    task,
    project_dim=None,
    task_type="regression",
):
    Y = torch.load(f"data/pred_gen/{dataset_name}/{tl_model_name}/Y_{task}.pt")
    num_examples = Y.shape[0]
    path = f"data/embeddings/{baseline_model_name}/{dataset_name}/{tl_model_name}_L=256/embeddings.pt"
    embeddings = torch.load(path)  # [dataset_size, embedding_dim]
    embeddings = embeddings[:num_examples, :]  # take only as many examples as in Y
    embeddings, Y, num_valid_examples = filter_valid_examples(embeddings, Y)
    print(f"Num valid examples: {num_valid_examples} / {num_examples}", file=sys.stderr)

    if project_dim is not None:
        embeddings = project(embeddings, project_dim)

    X_train, X_test, Y_train, Y_test = train_test_split(
        embeddings, Y, test_size=0.2, random_state=42
    )

    estimator = fit_estimator(X_train, Y_train, task_type)

    Y_pred = estimator.predict(X_test)
    relative_error = (
        compute_relative_error(Y_pred, Y_test) if task_type == "regression" else -1.0
    )
    score = estimator.score(X_test, Y_test)
    print(
        f"Baseline Model: {baseline_model_name}, rel_err: {relative_error:.2f}%, R^2: {score:.4f}"
    )
    return relative_error, score


def compute_relative_error(Y_pred: torch.Tensor, Y: torch.Tensor) -> float:
    relative_error = torch.mean(torch.abs(Y - Y_pred) / (Y + 1e-10)).item() * 100
    return relative_error


def create_dataset_baseline(
    model: SentenceTransformer,
    dataset_name: str,
    tl_model_name: str,
    dataset_size: int = int(1e4),
    L_max: int = 256,
    batch_size: int = 32,
) -> torch.Tensor:
    model = model.to("cuda")

    dataset = load_from_disk(
        f"data/preprocessed/{dataset_name}/{tl_model_name}_L={L_max}"
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.iter(batch_size=batch_size)
    embeddings_list = []

    for i in range(dataset_size // batch_size):
        conversations = (next(dataset))["conversation"]
        texts = convert_conversations_to_texts(conversations)

        embeddings = model.encode(
            texts, convert_to_tensor=True, device="cuda", show_progress_bar=False
        )  # [batch, embedding_dim]

        embeddings_list.append(embeddings.cpu())

        print(
            f"Processed embeddings batch {i + 1}/{dataset_size // batch_size}",
            file=sys.stderr,
        )

    all_embds = torch.cat(embeddings_list, dim=0)  # [dataset_size, embedding_dim]
    return all_embds


def convert_conversations_to_texts(conversations: list[list[dict]]) -> list[str]:
    texts = []
    for conv in conversations:
        text = ""
        for msg in conv:
            text += msg["content"] + "\n\n"
        texts.append(text)
    return texts


def create_datasets(
    model: PreTrainedModel,
    model_name: str,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    pooling_strategies: list[str] = ["last"],
    dataset_size: int = int(1e4),
    batch_size: int = 32,
    max_new_tokens: int = 1024,
    L_max: int = 256,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Creates dict of tensors X of residual stream activation for every layer for every pooling strategy in pooling_strategies
        - X[pooling_strategy][layer_idx] is tensor of activations for given layer and pooling strategy of shape [dataset_size, d_model]
    Creates dict of tensors Y with keys the different tasks
        - "pred_perplexity": Y["pred_perplexity"] is perplexity tensor
        - "pred_tokens_mass": Y["pred_tokens_mass"] is -log token mass tensor for given token ids
        - "pred_lengths": Y["pred_lengths"] is response lengths tensor

    Assumes dataset was preprocessed and has "input_ids" key which is tokenized input in chat format, and input_ids length is bounded by some L_max.
    Preprocessd dataset should be in data/preprocessed/{dataset_name}/{model_name}_L={L_max}
    """

    # workaround for HF Transformers output_hidden_states=True caching the final layer hidden state after the final LN which is not a part of the residual stream
    # Norm module is the final LN in modern HF transformers models see Qwen2Model for example https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L340
    # if not hasattr(model.model, "norm"):
    #     raise ValueError("HF WORKAROUND: Could not find final norm module in model")
    # identity = Identity()
    # final_ln_module = model.model.norm

    # load dataset
    dataset = load_from_disk(f"data/preprocessed/{dataset_name}/{model_name}_L={L_max}")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids"]]
    )  # keep only input_ids for batching
    dataset = dataset.shuffle(seed=42)
    tokenizer.padding_side = "left"  # so the last act corresponds to the last token
    tokenizer.pad_token = tokenizer.eos_token  # in case pad token is not defined
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    data_loader_iter = iter(dataloader)

    dataset_size = min(len(dataset), dataset_size)
    num_iterations = dataset_size // batch_size

    Ys = defaultdict(list)  # label_name -> label_values
    acts_cache = defaultdict(
        lambda: defaultdict(list)
    )  # pooling_strategy -> layer_idx -> Tensor
    for _ in tqdm(range(num_iterations), desc="Batches"):
        inputs = next(data_loader_iter)
        inputs = {key: val.to("cuda") for key, val in inputs.items()}

        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        # generation_output["hidden_states"] has the shape [generation_length/num forward passes, num_layers + 1, batch, seq_len, hidden_dim]
        # generation_length/num forward passes, num_layers + 1 are tuples
        # first forward pass has seq_len == prompt_len, subsequent forward passes has seq_len == 1 for the decoding stage
        # There is num_layers + 1 because the first hidden states are after the embedding layer
        # Consider not taking last hidden state becuase it is after the final layer norm

        # We care only for the prompt's hidden states
        hidden_states = generation_output["hidden_states"][0]
        # discard embeddins layer hidden states
        hidden_states = hidden_states[1:]

        # record acts
        for pooling_strategy in pooling_strategies:
            for layer_idx in range(model.config.num_hidden_layers):
                acts = hidden_states[layer_idx]  # [batch, seq_len, d_model]
                to_save = pool_activations(
                    acts, inputs["attention_mask"], pooling_strategy
                )  # [batch, d_model]
                acts_cache[pooling_strategy][layer_idx].append(to_save.cpu())
        prompt_len = inputs["input_ids"].shape[1]
        responses = generation_output["sequences"][
            :, prompt_len:
        ]  # [batch, max_response_len]

        Ys["response_token_ids"].append(responses.cpu())

        # record length
        lengths = compute_response_length(responses, tokenizer.eos_token_id)  # [batch]
        Ys["lengths"].append(lengths.cpu())

    # concatenate batches
    Xs_out = defaultdict(dict)
    for strat in pooling_strategies:
        for layer_idx in range(model.config.num_hidden_layers):
            Xs_out[strat][layer_idx] = torch.cat(acts_cache[strat][layer_idx], dim=0)

    Ys_out = {}
    for label_name in Ys.keys():
        if label_name == "response_token_ids":
            Ys_out[label_name] = concat_tensors_of_different_lengths(
                Ys[label_name], padding_value=tokenizer.eos_token_id
            )
        else:
            Ys_out[label_name] = torch.cat(Ys[label_name], dim=0)

    return Xs_out, Ys_out


def pool_activations(
    acts: torch.Tensor, att_mask: torch.Tensor, pooling_strategy: str
) -> torch.Tensor:
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
        masked_acts = acts * att_mask.unsqueeze(-1)  # [batch, seq_len, d_model]
        sums = masked_acts.sum(dim=1)  # sum over sequence dim [batch, d_model]
        counts = att_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        pooled_acts = sums / counts  # [batch, d_model]
    elif pooling_strategy == "max":
        mask = att_mask.unsqueeze(-1).bool()  # [batch, seq_len, 1]
        acts_masked = acts.float().masked_fill(
            ~mask, float("-inf")
        )  # [batch, seq_len, d_model]
        pooled_acts = torch.max(
            acts_masked, dim=1
        ).values  # max over sequence dim [batch, d_model]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return pooled_acts


def record_tokens_mass(logits: torch.Tensor, tokens_ids: list[int]) -> torch.Tensor:
    last_logits = logits[:, -1, :]  # [batch, vocab_size]
    probs = torch.nn.functional.softmax(last_logits, dim=-1)  # [batch, vocab_size]
    tokens_mass = probs[:, tokens_ids].sum(dim=-1)  # [batch]
    mlog_mass = -torch.log(tokens_mass + 1e-10)  # [batch]
    return mlog_mass


def record_nll(
    tokens: torch.Tensor, att_mask: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    # tragets are all tokens except first
    target_tokens = tokens[:, 1:]  # [batch, seq_len - 1]

    # we take log prob on all tokens except last
    minus_log_probs = (-torch.nn.functional.log_softmax(logits, dim=-1))[
        :, :-1, :
    ]  #  [batch, seq_len - 1, vocab_size]
    target_log_probs = torch.gather(
        minus_log_probs, dim=-1, index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len - 1]

    att_mask = att_mask[
        :, :-1
    ]  # adjust attention mask on the tokens we measured log probs for [batch, seq_len - 1]
    masked_target_log_probs = target_log_probs * att_mask  # [batch, seq_len - 1],
    sums = masked_target_log_probs.sum(dim=-1)  # [batch]
    counts = att_mask.sum(dim=-1).clamp(min=1)  # [batch]
    mean_nll = sums / counts  # [batch]
    return mean_nll


def compute_response_length(responses: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    # responses: [batch, gen_len]
    not_eos = responses != eos_token_id
    lengths = not_eos.sum(dim=-1) + 1  # +1 to account for last eos token; [batch]
    is_truncated = responses[:, -1] != eos_token_id  # [batch]
    lengths[is_truncated] = -1  # mark truncated generations with -1
    return lengths


def concat_tensors_of_different_lengths(
    tensors: list[torch.Tensor], padding_value: int
) -> torch.Tensor:
    num_rows = sum(tensor.size(0) for tensor in tensors)
    max_cols = max(tensor.size(1) for tensor in tensors)
    dtype = tensors[0].dtype
    device = tensors[0].device
    result = torch.full((num_rows, max_cols), padding_value, dtype=dtype, device=device)
    offset = 0
    for tensor in tensors:
        rows, cols = tensor.size()
        result[offset : offset + rows, :cols] = tensor
        offset += rows
    return result


def project(X: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Projects X to target_dim using radnom gaussian matrix.
    X: [num_samples, original_dim]
    Returns: X_projected: [num_samples, target_dim]
    """
    original_dim = X.shape[1]
    torch.manual_seed(42)  # for reproducibility
    projection_matrix = (target_dim**-0.5) * torch.randn(original_dim, target_dim)

    X_projected = X @ projection_matrix  # [num_samples, target_dim]
    return X_projected


def filter_valid_examples(
    X: torch.Tensor, Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filters out examples where Y is -1.
    X: [num_samples, dim]
    Y: [num_samples]
    Returns: X_filtered: [num_valid_samples, dim], Y_filtered: [num_valid_samples]
    """
    valid_mask = Y != -1
    X_filtered = X[valid_mask]
    Y_filtered = Y[valid_mask]
    num_valid_examples = valid_mask.sum().item()
    return X_filtered, Y_filtered, num_valid_examples
