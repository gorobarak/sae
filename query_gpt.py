import openai
import os

with open("openai_key.txt", "r") as f:
    openai_key = f.read().strip()
os.environ["OPENAI_API_KEY"] = openai_key

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


