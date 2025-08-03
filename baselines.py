import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import os
import sys
from utils import class_idx_to_class_name_dbpedia

# pooling across the sequence dimension
def mean_pooling(embeddings, attention_mask, max_seq_length=128):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    if max_seq_length is not None:
        input_mask_expanded = input_mask_expanded[:, :max_seq_length, :]
        embeddings = embeddings[:, :max_seq_length, :]
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) # divide by the number of non-padding tokens in the sequence

def create_dense_representation(dataset="fancyzhx/dbpedia_14", embedding_model="sentence-transformers/all-mpnet-base-v2", num_classes=7):
    """
    Create a dense representation of the classes in the dataset using the specified embedding model.
    
    Args:
        dataset: The dataset to be embedded.
        embedding_model: The model used for generating embeddings.
        num_classes: The number of classes in the dataset.

    Returns:
        None: The function saves the dense representations to disk.
    """
    NUMBER_OF_EXAMPLES_IN_CLASS = 40000 
    batch_size = 256
    DTYPE = torch.float32
    DEVICE = "cuda"

    # Load the dataset tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.to(DEVICE); model.to(DTYPE)
    dataset = load_dataset(dataset, split='train')


    
    for i in range(num_classes):
        dense_representation_acc = torch.zeros((model.config.hidden_size), dtype=DTYPE, device=DEVICE)
        offset = i * NUMBER_OF_EXAMPLES_IN_CLASS
        for j in range(offset, offset + NUMBER_OF_EXAMPLES_IN_CLASS, batch_size):
           
            # get the current batch
            interval_end = min(j + batch_size, offset + NUMBER_OF_EXAMPLES_IN_CLASS)
            batch = dataset[j: interval_end]
            batch_texts = batch['content'] # [batch]
            
            # tokenize the batch
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            # encoded input = (input_ids, attention_mask)
            encoded_input = {key: value.to(DEVICE) for key, value in encoded_input.items()} # [batch, seq_len]

            # get model embeddings
            with torch.no_grad():
                embeddings = model(**encoded_input)[0] # [batch, seq_len, d_model]

            # mean across sequence dimension
            pooled_embeddings = mean_pooling(embeddings, encoded_input['attention_mask'], max_seq_length=128) #  [batch, d_model]
            assert pooled_embeddings.shape[0] == (interval_end - j)
            assert pooled_embeddings.shape[1] == model.config.hidden_size
            assert len(pooled_embeddings.shape) == 2 

            # sum across batch dimension
            pooled_embeddings = pooled_embeddings.sum(dim=0) # [d_model]
            assert pooled_embeddings.shape[0] == model.config.hidden_size
            assert len(pooled_embeddings.shape) == 1
            dense_representation_acc += pooled_embeddings

        dense_representation_acc /= NUMBER_OF_EXAMPLES_IN_CLASS
    
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{class_idx_to_class_name_dbpedia(i)}"
        os.makedirs(dir_path, exist_ok=True)
        file_name = "dense_representation.pt"
        torch.save(dense_representation_acc.cpu(), os.path.join(dir_path, file_name))
        print(f"Saved dense representation for class {class_idx_to_class_name_dbpedia(i)}", file=sys.stderr)



def dense_representation_concept_ranking(concepts, embedding_model="sentence-transformers/all-mpnet-base-v2", num_classes=7):
    # load embedding model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.to("cuda"); model.to(torch.float32)
    
    lines = []

    for i in range(num_classes):
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{class_idx_to_class_name_dbpedia(i)}"
        file_name = "dense_representation.pt"
        dense_representation = torch.load(os.path.join(dir_path, file_name))
        dense_representation = dense_representation.cuda()

        concept_scores = []
        for concept in concepts:
            # tokenize the concept
            encoded_input = tokenizer([concept], padding=True, truncation=True, return_tensors="pt")
            encoded_input = {key: value.to("cuda") for key, value in encoded_input.items()}

            # get model embeddings
            with torch.no_grad():
                embeddings = model(**encoded_input)[0] # [batch, seq_len, d_model]

            # mean across sequence dimension
            pooled_embeddings = mean_pooling(embeddings, encoded_input['attention_mask']) # [batch, d_model]
            assert pooled_embeddings.shape[0] == 1 # batch is 1
            assert pooled_embeddings.shape[1] == model.config.hidden_size

            concept_embedding = pooled_embeddings[0]  # [d_model]
            concept_scores.append(F.cosine_similarity(dense_representation, concept_embedding, dim=0).item())

        concept_scores = torch.tensor(concept_scores)
        sorted_indices = torch.argsort(concept_scores, descending=True)
        sorted_concepts = [concepts[i] for i in sorted_indices]
        sorted_scores = concept_scores[sorted_indices]

        # write ranking
        lines.append(f"Ranking of concepts for class {class_idx_to_class_name_dbpedia(i)}:\n")
        j = 1
        for concept, score in zip(sorted_concepts, sorted_scores):
            lines.append(f"{j}.{concept}: {score.item():.3f}\n")
            j += 1
        lines.append("\n")
    
    # write to file
    filename = "concept_ranking_generated_topics.txt"
    dir_path = "checkpoints/dense_representations"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), "w") as f:
        f.writelines(lines)



        


