import openai
import os
import sys
import requests 
import pdb

with open("tokens/openai_key_personal.txt", "r") as f:
    openai_key = f.read().strip()
os.environ["OPENAI_API_KEY"] = openai_key

def query_gpt(prompt, model="gpt-4.1-nano"):
    client = openai.OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt
    )
    text = response.output_text.strip()
    return text


def query_explainer_model(topk_samples,  model="gpt-4o-mini"):
    """
    Queries the OpenAI API to get explanations for the top-k samples.
    """
    client = openai.OpenAI()

    topk_samples = standerdize(topk_samples)
    
    prompt = construct_prompt(topk_samples)
    response = client.responses.create(
        model=model,
        input=prompt
    )
    description = response.output_text.strip()
    return description


"""
Assumes samples are sorted in descending order
"""
def construct_prompt(topk_samples):
    prefix="""
You are given the top 10 text samples that most strongly activated a specific feature in a Sparse Autoencoder (SAE) trained on natural language. Higher activation values indicate stronger activation.

**Your task**:
Write **one short, clear sentence** that describes the common theme, pattern, or concept captured by this feature—one that explains why it activated across these specific samples.

Samples (in the format of ``<index>.  <activation_value> -- <text>``):
"""
    prompt_parts = [prefix]


    postfix="""
Output format:
One short sentence summarizing what this feature detects in text. No need to explain the activation values or individual samples. No need to prefix the sentence with "This feature detects" or similar phrases. **Just provide the description directly.**
"""
    for i, sample in enumerate(topk_samples):
        act_value = sample[0]
        text = sample[1]
        prompt_parts.append(f"{i+1}. {act_value:.1f} -- {text}")
    
    prompt_parts.append(postfix)
    prompt =  "\n".join(prompt_parts)
    return prompt

def standerdize(topk_samples):
    topk_samples.sort(reverse=True)
    #act_values = [sample[0] for sample in topk_samples]
    # max_act = act_values[0]
    # min_act = act_values[-1]
    # standrdize_values = [((act_val - min_act ) / (max_act - min_act + 1e-3))*10 for act_val in act_values]
    # standrdize_topk =[]
    # for i in range(len(topk_samples)):
    #     standrdize_topk.append((standrdize_values[i], topk_samples[i][2]))
    # return standrdize_topk
    return [(sample[0], sample[2]) for sample in topk_samples]


def build_descriminate_task_prompt(real_text, decoy_texts, top10_real, real_idx, hook_point):
    """
    Builds a prompt for the descriminate task.
    """
    layer = hook_point.split(".")[1]  
    descriptions = []
    act_vals = top10_real.values
    model_id = "gemma-2-2b"
    release_and_layer = f"{layer}-gemmascope-res-16k" 
    for feature_idx in top10_real.indices:
        r = requests.get(
            f"https://www.neuronpedia.org/api/feature/{model_id}/{release_and_layer}/{feature_idx}"
            )
        if r.status_code == 200:
            body = r.json()
            explainations = body["explanations"]
            descriptions.append(explainations[-1]["description"])
        else:
            print(f"Failed to fetch explanations: {r.status_code} - {r.reason}", file=sys.stderr)

    num_of_texts = len(decoy_texts) + 1

    prompt = f"""
Your task is to determine which of the {num_of_texts} provided texts most accurately corresponds to the given set of text features, derived from a Sparse Autoencoder (SAE). The features listed below are sorted in descending order by activation value, representing their strength and relevance to the original text.

The features list sorted by activation value:

1. {act_vals[0]} - {descriptions[0]}
2. {act_vals[1]} - {descriptions[1]}
3. {act_vals[2]} - {descriptions[2]}
4. {act_vals[3]} - {descriptions[3]}
5. {act_vals[4]} - {descriptions[4]}
6. {act_vals[5]} - {descriptions[5]}
7. {act_vals[6]} - {descriptions[6]}
8. {act_vals[7]} - {descriptions[7]}
9. {act_vals[8]} - {descriptions[8]}
10. {act_vals[9]} - {descriptions[9]}

The texts:

{build_texts_order(real_text, decoy_texts, real_idx)}

Based solely on the listed features, identify which of these texts, when decomposed by the SAE, would produce the given feature activations. Respond with the number of the text that best matches these features. Provide no explanations, prefixes, or additional context—your response should consist of only a single number.
    """
    return prompt

def build_texts_order(real_text, decoy_texts, real_idx):
    """
    Builds the texts in the order specified by real_idx.
    """
    length = len(decoy_texts) + 1
    prompt_parts = []
    iter_decoy_texts = iter(decoy_texts)
    for i in range(1, length + 1):
        if i == real_idx:
            prompt_parts.append(f"{i}. {real_text}")
        else:
            prompt_parts.append(f"{i}. {next(iter_decoy_texts)}")

    return "\n".join(prompt_parts)