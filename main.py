from classifier import LinearClassifier
from utils import class_idx_to_class_name_dbpedia, print_and_write, get_file_prefix
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_default_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch
from sae_lens import SAE
from my_activation_store import UnsupervisedActivationStore, SupervisedActivationsStore
from datasets import load_dataset
import os
from query_gpt import query_gpt, build_descriminate_task_prompt
import sys
import random
from neuropedia import get_description



def generate_population_level_insights(num_dbpedia_classes=7, reactivations=False, layer=18, matrix_multiply=False):
    
    assert layer in range(0,26), "layer must be in range 0-25"
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]
    files_prefix = get_file_prefix(reactivations=reactivations, matrix_multiply=matrix_multiply)
    lines = []
    
    for class_idx in range(num_dbpedia_classes):
        class_name = class_idx_to_class_name_dbpedia(class_idx)
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{layer}", f"dbpedia_class_{class_idx}")
        
        # load activations/reactivations
        if reactivations:
            with open(os.path.join(dir_path, "reactivations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU
        else:
             with open(os.path.join(dir_path, "activations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU
                    
        
        # aggregate activations along sequence and batch dimensions without bos token
        agg_activation = activations[:, 1:, :].mean(dim=(0, 1)) # [d_model]
        
        
        # get dictionary activation
        agg_activation = agg_activation.to('cuda')  # Move to GPU before passing into SAE
        with torch.no_grad():
            if matrix_multiply:
                dict_activations = agg_activation @ sae.W_enc  # [d_sae]
            else:
                dict_activations = sae.encode(agg_activation.unsqueeze(0)).squeeze(0) # [d_sae]


        # get top-k features
        dict_topk = torch.topk(dict_activations, k=15, sorted=True)

        # Save topk features indices and values
        filename = files_prefix + "features_indicies.pt"
        with open(os.path.join(dir_path, filename), "wb") as f:
            torch.save(dict_topk.indices, f)
        filename = files_prefix + "features_values.pt"
        with open(os.path.join(dir_path, filename), "wb") as f:
            torch.save(dict_topk.values, f)

        # get descriptions for top-k features
        model_id = "gemma-2-2b"
        neuropedia_sae_id = f"{layer}-gemmascope-res-16k"
        lines = []
        print_and_write(f"Top features for class {class_name}", lines)
        for feature_idx, act_val in zip(dict_topk.indices, dict_topk.values):
            if act_val == 0.0:
                break
            description = get_description(feature_idx, model_id=model_id, neuronpedia_sae_id=neuropedia_sae_id)
            if description:
                print_and_write(f"Feature {feature_idx}:", lines)
                print_and_write(f"{act_val.item()} -- {description}", lines)
            

    # write to file
    filename = files_prefix + "PLI.txt"
    with open(os.path.join("checkpoints",f"{model_name}_layer_{layer}" ,filename), "w") as f:
        f.writelines(lines)

def create_histogram_for_population_level_insights(num_dbpedia_classes=7, reactivations=False, layer=18, matrix_multiply=False):
    assert layer in range(0,26), "layer must be in range 0-25"
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]
    files_prefix = get_file_prefix(reactivations=reactivations, matrix_multiply=matrix_multiply)

    for class_idx in range(num_dbpedia_classes):
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{layer}", f"dbpedia_class_{class_idx}")

        # load activations/reactivations
        if reactivations:
            with open(os.path.join(dir_path, "reactivations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU
        else:
            with open(os.path.join(dir_path, "activations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU
        
        # Filter bos token
        activations = activations[:, 1:, :]  # [batch, seq_len - 1, d_model]

        # Aggregate along sequence dimention
        activations = activations.mean(dim=1) # [batch, d_model]
        assert len(activations.shape) == 2, "Activations should be of shape [batch, d_model]"
        assert activations.shape[0] == 40000, "Activations should have 40000 samples per class"
        assert activations.shape[1] == 2304, "Activations should have 2304 features per sample"

        # Get dictionaries activations
        activations = activations.to('cuda')  # Move to GPU before passing into SAE
        with torch.no_grad():
            if matrix_multiply:
                dict_activations = activations @ sae.W_enc # [batch, d_sae]
        
        # Sum along batch dimension to get count per dictionary feature
        dict_activations_bool = dict_activations > 0.0
        histogram = dict_activations_bool.long().sum(dim=0)  # [d_sae]

        # Save histogram
        filename = files_prefix + "histogram.pt"
        with open(os.path.join(dir_path, filename), "wb") as f:
            torch.save(histogram, f)
        print(f"Histogram saved to {os.path.join(dir_path, filename)}", file=sys.stderr)

def descriminate_task(use_reactivations=False, real_class_idx=2, decoy_class_idx=2, use_same_class_as_decoy=False, num_of_decoy_classes=25):
    if use_same_class_as_decoy:
        decoy_class_idx = real_class_idx
    NUM_SAMPLES_IN_CLASS = 40000
    num_of_tests = 250
    offset_real = real_class_idx * NUM_SAMPLES_IN_CLASS
    offset_decoy = decoy_class_idx * NUM_SAMPLES_IN_CLASS
    model_name = "gemma-2-2b"
    hook_point = "blocks.24.hook_resid_post"
    real_class_path = f"checkpoints/dbpedia_class_{real_class_idx}_{model_name}_{hook_point}"
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float32)
    tokenizer = model.tokenizer
    dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
    sae_release= "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_24/width_16k/canonical" 
    sae, sae_cfg, _ = SAE.from_pretrained(sae_release, sae_id, device="cuda")
    random.seed(42)  # For reproducibility
    
    # Load activations or reactivations
    if use_reactivations:
        with open(os.path.join(real_class_path, "reactivations.pt"), "rb") as f:
            real_activations = torch.load(f).to('cuda')

    else:
        with open(os.path.join(real_class_path, "activations.pt"), "rb") as f:
            real_activations = torch.load(f).to('cuda')
    
    correct = 0
    for i in range(num_of_tests):
        
        indexes = random.sample(range(NUM_SAMPLES_IN_CLASS), num_of_decoy_classes + 1)
        
        real_example_idx = indexes[0]  
        real_act = real_activations[real_example_idx] # [seq_len, d_model]
        real_text = dataset[real_example_idx + offset_real]['content']
        real_text = tokenizer.decode(model.to_tokens(real_text, prepend_bos=True, truncate=True).squeeze(0)[:128])

        decoy_texts = []
        for idx in indexes[1:]:
            decoy_text = dataset[idx + offset_decoy]['content']
            decoy_text = tokenizer.decode(model.to_tokens(decoy_text, prepend_bos=True, truncate=True).squeeze(0)[:128])
            decoy_texts.append(decoy_text)

        
        dict_real_acts = sae.encode(real_act.unsqueeze(0)).squeeze(0)  # [seq_len, d_sae]
        
        agg_dict_acts = dict_real_acts.max(dim=0).values # [d_sae] / aggregate dict activations 

        top10_real = torch.topk(agg_dict_acts, k=10, sorted=True)

        real_idx_amongst_decoys = random.sample(range(num_of_decoy_classes + 1), 1)[0] + 1 # 1, 2, ...., num_of_decoy_classes + 1
    
        prompt = build_descriminate_task_prompt(real_text, decoy_texts, top10_real, real_idx_amongst_decoys, hook_point)

        response = query_gpt(prompt, model="gpt-4o-mini")
        response = response.strip().lower()
        try:
            response = int(response)
        except ValueError:
            print(f"Error: {response} isn't a valid response", file=sys.stderr)
            continue
        if response == real_idx_amongst_decoys:
            correct += 1
        

    accuracy = correct / num_of_tests
    
    filename = ("reactivations" if use_reactivations else "activations") + f"_num_of_decoys_{num_of_decoy_classes}" + "_descriminate_task_results.txt"
    if use_same_class_as_decoy:
        filename = "same_class_decoy_" + filename
    
    with open(os.path.join(real_class_path, filename), "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Total tests: {num_of_tests}\n")

     

if __name__ == "__main__":
    
    # descriminate_task(use_reactivations=False, real_class_idx=5, decoy_class_idx=2, use_same_class_as_decoy=True, num_of_decoy_classes=5)
    
    for bool_val in [True, False]:
        create_histogram_for_population_level_insights(num_dbpedia_classes=7, reactivations=bool_val, layer=24, matrix_multiply=True)

    generate_population_level_insights(num_dbpedia_classes=7, reactivations=True, layer=18, matrix_multiply=True)
    