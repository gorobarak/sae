from re import M
import torch
import torch.nn.functional as F
import torch.distributions as distributions
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import sys
from utils import class_idx_to_class_name_dbpedia, get_file_suffix
NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA = 40000 


def create_representations_for_classes(dataset="fancyzhx/dbpedia_14", num_classes=7):
    """
    Create a dense representation of the classes in the dataset using the specified embedding model.
    
    Args:
        dataset: The dataset to be embedded.
        embedding_model: The model used for generating embeddings.
        num_classes: The number of classes in the dataset.

    Returns:
        None: The function saves the dense representations to disk.
    """
    batch_size = 256
    DTYPE = torch.float32
    DEVICE = "cuda"

    # Load the dataset and model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to(DEVICE); model.to(DTYPE)
    dataset = load_dataset(dataset, split='train')

    for i in range(num_classes):
        dense_representation_acc = torch.zeros((model.get_sentence_embedding_dimension()), dtype=DTYPE, device=DEVICE)
        offset = i * NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA
        for j in range(offset, offset + NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA, batch_size):
           
            # get the current batch
            interval_end = min(j + batch_size, offset + NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA)
            batch = dataset[j: interval_end]
            batch_texts = batch['content'] # [batch]
            
            # get model embeddings
            embeddings = model.encode(batch_texts) # [batch, d_model]
            embeddings = torch.tensor(embeddings, device=DEVICE)
            assert embeddings.shape[0] == len(batch_texts)
            assert embeddings.shape[1] == model.get_sentence_embedding_dimension()
            
            # sum across batch dimension
            embeddings = embeddings.sum(dim=0) # [d_model]
            assert embeddings.shape[0] == model.get_sentence_embedding_dimension()
            assert len(embeddings.shape) == 1
            dense_representation_acc += embeddings

        # Divide by number of samples in class to get mean
        dense_representation_acc /= NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA
    
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{i}"
        os.makedirs(dir_path, exist_ok=True)
        file_name = "dense_representation.pt"
        torch.save(dense_representation_acc.cpu(), os.path.join(dir_path, file_name))
        print(f"Saved dense representation for class {class_idx_to_class_name_dbpedia(i)}", file=sys.stderr)



def concept_ranking_whole_classes(concepts, 
                                         private=True,
                                         epsilon=0.1,
                                         num_classes=7):
    
    # load embedding model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to("cuda"); model.to(torch.float32)
    
    lines = []
    top_1_correct = 0
    top_3_correct = 0
    for i in range(num_classes):
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{i}"
        file_name = "dense_representation.pt"
        dense_representation = torch.load(os.path.join(dir_path, file_name))
        dense_representation = dense_representation.cuda()

        if private:
           make_private_embedding(dense_representation, 
                                  num_of_users_in_database=NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA, 
                                  epsilon=epsilon)
        concept_scores = []
        for concept in concepts:
            # get concept embeddings
            concept_embedding = model.encode(concept) #[d_model]
            concept_embedding = torch.tensor(concept_embedding, device="cuda")
            concept_scores.append(F.cosine_similarity(dense_representation, concept_embedding, dim=0).item())

        concept_scores = torch.tensor(concept_scores)
        sorted_indices = torch.argsort(concept_scores, descending=True)
        sorted_concepts = [concepts[i] for i in sorted_indices]
        sorted_scores = concept_scores[sorted_indices]

        # update scores
        if sorted_concepts[0].lower() == class_idx_to_class_name_dbpedia(i).lower():
            top_1_correct += 1
        if class_idx_to_class_name_dbpedia(i) in [c.lower() for c in sorted_concepts[:3]]:
            top_3_correct += 1


        # write ranking
        lines.append(f"Ranking of concepts for class {class_idx_to_class_name_dbpedia(i)}:\n")
        j = 1
        for concept, score in zip(sorted_concepts, sorted_scores):
            lines.append(f"{j}.{concept}: {score.item():.3f}\n")
            j += 1
        lines.append("\n")
    
    # write to file
    suffix = get_file_suffix(private=private)
    if private:
        suffix += f"_epsilon={epsilon}"
    
    # concept ranking
    filename = "concept_ranking" + suffix + ".txt"
    dir_path = "checkpoints/dense_representations"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), "w") as f:
        f.writelines(lines)
    print("written concept ranking to", os.path.join(dir_path, filename), file=sys.stderr)

    # score
    top_1_score = top_1_correct / num_classes
    top_3_score = top_3_correct / num_classes
    print(f"Top-1 accuracy: {top_1_score:.3f}, Top-3 accuracy: {top_3_score:.3f}", file=sys.stderr)
    return top_1_score, top_3_score





def concept_ranking(text_batch, concepts):
    pass

def make_private_embedding(embedding_vec, num_of_users_in_database, epsilon):
    # normalize the vector 
    normalized_embedding_vec = F.normalize(embedding_vec, p=2, dim=0)
    # add laplace noise to each coordinate
    d = normalized_embedding_vec.shape[0]
    scale = d / (num_of_users_in_database * epsilon)
    laplace_dist = distributions.Laplace(loc=0, scale=scale)
    noise = laplace_dist.sample(embedding_vec.shape)
    noise = noise.to(embedding_vec.device)
    embedding_vec += noise


