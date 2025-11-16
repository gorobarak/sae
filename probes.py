from collections import defaultdict
import sys
from sympy import per
from transformer_lens.hook_points import HookedRootModule
from datasets import IterableDataset
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def create_dataset_tokens_mass(
        model: HookedRootModule,
        dataset: IterableDataset,
        hookpoint: str,
        tokens_ids: list[int],
        dataset_size: int = int(1e3)
        ):
    """
    Creates a tensor X, Y where X is a tensor of activations of shape [dataset_size, seq_len, hidden_dim]
    and Y is a tensor of the tokens ids minus log probabily mass [dataset_size, seq_len, 1]
    The function assumes the dataset is dataset of conversations with "conversation" key.
    """
    model = model.to("cuda")
    seq_len = 128
    batch_size = 64
    dataset = dataset.iter(batch_size)
    acts_list = []
    mlogprob_list = []

    for i in range(dataset_size // batch_size):
        
        conversations = (next(dataset))["conversation"]
        tokens = model.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
            
        ).to("cuda")  # [batch, seq_len]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=[hookpoint]
            )
        
        acts = cache[hookpoint]  # [batch, seq_len, d_model]
        acts_list.append(acts.cpu())

        # logits: [batch, seq_len, vocab_size]
        probs = torch.softmax(logits, dim=-1) # [batch, seq_len, vocab_size]
        tokens_probs = probs[:, :, tokens_ids] # [batch, seq_len, len(tokens_ids)]
        mass = tokens_probs.sum(dim=-1, keepdim=True) # [batch, seq_len, 1]
        mlogprob = -torch.log(mass + 1e-10) # [batch, seq_len, 1]
        mlogprob_list.append(mlogprob.cpu())

        print(f"Processed batch {i+1}/{dataset_size // batch_size}", file=sys.stderr)
    
    X = torch.cat(acts_list, dim=0)  # [dataset_size, seq_len, hidden_dim]
    Y = torch.cat(mlogprob_list, dim=0)  # [dataset_size, seq_len, 1]

    return X, Y



def fit_ridge_regression(X: torch.Tensor, Y: torch.Tensor):

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, Y)

    return ridge



def eval(model_name):
    path = f"data/pred_tokens_density/{model_name}"
    hookpoints = os.listdir(path)
    hookpoints = sorted(hookpoints, key=lambda x: int(x.split(".")[1]))
    layers = [int(hookpoint.split(".")[1]) for hookpoint in hookpoints]
    scores = []
    for hookpoint in hookpoints:
        X = torch.load(f"{path}/{hookpoint}/X.pt")
        Y = torch.load(f"{path}/{hookpoint}/Y.pt")


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train = X_train.reshape(-1, X_train.shape[-1])
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        ridge = fit_ridge_regression(X_train, Y_train)

        test_score = ridge.score(X_test, Y_test)

        print(f"Model: {model_name}, Hookpoint: {hookpoint}, Test R^2: {test_score:.4f}")
        scores.append(test_score)

    return layers, scores


def create_dataset_perplexity(
        model: HookedRootModule,
        tokenizer: AutoTokenizer,
        dataset: IterableDataset,
        dataset_size: int = int(1e3)
        ):
    """
    Creates a dataset for every layer of last activations Xs and perplexity Y.
    Assumes dataset has "conversation" key which is in chat format
    """
    model = model.to("cuda")
    perplexity_list = []
    acts_list_per_layer = defaultdict(list)
    dataset_iter = iter(dataset)
    hookpoints = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]

    for i in range(dataset_size):
        conv = (next(dataset_iter))["conversation"]
        tokens = tokenizer.apply_chat_template(conv, padding=False, return_tensors="pt", tokenize=True).to("cuda")  # [1, seq_len]
        
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=hookpoints)
        
        # save acts
        for hookpoint in cache.keys():
            acts = cache[hookpoint]  # [1, seq_len, d_model]
            acts_list_per_layer[hookpoint].append(acts[0, -1, :].cpu())  # [d_model]

        # calculate perplexity
        targets_tokens = tokens.clone()[:, 1:].squeeze() # shift left; [seq_len - 1]
        logits = logits[:, :-1, :].squeeze() # remove last token logits; [seq_len - 1, vocab_size]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [seq_len - 1, vocab_size]
        nll = torch.nn.functional.nll_loss(log_probs, targets_tokens, reduction='mean')
        perplexity = torch.exp(nll) # scalar
        perplexity_list.append(perplexity) 
        
        print(f"Processed example {i+1}/{dataset_size}", file=sys.stderr)
    
    Xs = []
    for hookpoint in hookpoints:
        acts_list = acts_list_per_layer[hookpoint]
        X = torch.stack(acts_list)  # [dataset_size, hidden_dim]
        Xs.append(X)
    Y = torch.stack(perplexity_list, dim=0)  # [dataset_size]

    return Xs, Y

def eval_perplexity_prediction(model_name):
    path = f"data/pred_perplexity/{model_name}"
    hookpoints = os.listdir(path)
    hookpoints = sorted(hookpoints, key=lambda x: int(x.split(".")[1]))
    layers = [int(hookpoint.split(".")[1]) for hookpoint in hookpoints]
    rel_errs = []
    r2_scores = []
    for hookpoint in hookpoints:
        X = torch.load(f"{path}/{hookpoint}/X.pt")
        Y = torch.load(f"{path}/{hookpoint}/Y.pt")


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_last_hidden_state = X_train[:, -1, :].squeeze()  # [num_samples, hidden_dim]
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test_last_hidden_state = X_test[:, -1, :].squeeze()  # [num_samples, hidden_dim]
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        ridge = fit_ridge_regression(X_train_last_hidden_state, Y_train)

        relative_error = compute_relative_error(ridge, X_test_last_hidden_state, Y_test)
        r2_score = ridge.score(X_test_last_hidden_state, Y_test)

        print(f"Model: {model_name}, Hookpoint: {hookpoint}, Test relative error: {relative_error:.4f}, R^2: {r2_score:.4f}")
        rel_errs.append(relative_error)
        r2_scores.append(r2_score)

    return layers, rel_errs, r2_scores

def compute_relative_error(ridge: Ridge, X_test: torch.Tensor, Y_test: torch.Tensor) -> float:
    Y_pred = ridge.predict(X_test)
    relative_error = torch.mean(torch.abs(Y_test - Y_pred) / (Y_test + 1e-10)).item()
    return relative_error

def create_baseline_dataset(model: SentenceTransformer, dataset: IterableDataset, dataset_size: int):
    
    model = model.to("cuda")
    batch_size = 50
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
        
        assert embeddings.size() == (batch_size, model.get_sentence_embedding_dimension())
        
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

def eval_baseline(llm_model_name: str):
    print("Evaluating baseline for perplexity prediction...")
    path = "data/all-mpnet-base-v2/lmsys-chat-1m/embeddings.pt"
    embeddings = torch.load(path)  # [dataset_size, embedding_dim]
    Y = torch.load(f"data/pred_perplexity/{llm_model_name}/blocks.0.hook_resid_post/Y.pt")

    X_train, X_test, Y_train, Y_test = train_test_split(embeddings, Y, test_size=0.2, random_state=42)

    ridge = fit_ridge_regression(X_train, Y_train)

    relative_error = compute_relative_error(ridge, X_test, Y_test)
    r2_score = ridge.score(X_test, Y_test)

    print(f"LLM: {llm_model_name}, relative error: {relative_error:.2f}, R^2: {r2_score:.2f}")

    return relative_error, r2_score