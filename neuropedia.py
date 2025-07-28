import requests
import sys

query_to_generate_topics = """You are tasked with generating a comprehensive list of high-level topics or concepts that can be used to categorize user queries to large language models (LLMs), such as ChatGPT. The purpose of this list is to analyze and monitor trending topics and user engagement patterns.

Instructions:

Create a list of broad, mutually exclusive topics or concepts. Each topic should be defined such that any given LLM query can be classified predominantly under a single topic (i.e., the topics should not overlap wherever possible).

The list should strive to be exhaustive, aiming to cover the entire range of queries that users might pose to an LLM. Consider all domains, use-cases, and user intents.

The topics should be general enough to accommodate a wide variety of specific questions, but granular enough to ensure that each topic is distinct from the rest.

Think carefully about edge cases—try to minimize ambiguity and ensure that every user query could reasonably fit under one, and only one, of the topics.

**Output format**:

List each topic as a bullet point. 
Don't provide information other than the topic names.

**Example**:

*Health & Medicine

Now, generate the complete list according to the instructions above."""
TOPICS = ["Health and Medicine",
        "Mathematics and Formal Logic",
        "Natural Sciences and Engineering",
        "Programming, Data and AI Development",
        "Consumer Tech Use and Troubleshooting",
        "Business, Finance and Economics",
        "Law, Policy and Government Services",
        "Education and Academic Support",
        "Career and Workplace",
        "Productivity and Personal Development",
        "Relationships, Parenting and Social Advice",
        "Food and Cooking",
        "Travel, Geography and Local Recommendations",
        "Arts, Culture and Humanities",
        "Entertainment and Pop Culture",
        "Sports and Physical Training",
        "Religion, Spirituality and Philosophy",
        "Home, DIY, Gardening and Pets",
        "Shopping and Product Recommendations",
        "Design, Visual Media and Content Production",
        "Creative Writing, Storytelling and Role-Play",
        "Language Learning and Translation",
        "Games, Puzzles and Recreational Math",
        "News and Current Events",
        "LLM Use, Prompting and AI Ethics",
        "Small Talk, Humor and Social Chat",
        "Miscellaneous or Unclassifiable"]

def get_description(feature_idx, model_id, neuronpedia_sae_id):
    r = requests.get(
    f"https://www.neuronpedia.org/api/feature/{model_id}/{neuronpedia_sae_id}/{feature_idx}"
    )
    if r.status_code == 200:
        body = r.json()
        try:
            explanation = body["explanations"][-1]
        except IndexError:
            print(f"feature {feature_idx} has no description", file=sys.stderr)
            return None
        return explanation["description"]
    else:
        print(f"Failed to fetch explanation: {r.status_code} - {r.reason}", file=sys.stderr)
        print(f"Request URL: https://www.neuronpedia.org/api/feature/{model_id}/{neuronpedia_sae_id}/{feature_idx}", file=sys.stderr)
        return None
    
# Concept to his top features 
# TODO: Does this returns the most relevant features for the concept?
def get_concept_feature_indicies(concept, model_id="gemma-2-2b", neuronpedia_sae_id="24-gemmascope-res-16k"):
    r = requests.post(
                "https://www.neuronpedia.org/api/explanation/search",
                headers={
                "Content-Type": "application/json"
                },
                json={
                "modelId": model_id,
                "layers": [
                    neuronpedia_sae_id
                ],
                "query": concept
                }
            )
    if r.status_code == 200:
        body = r.json()
        feature_indices = []
        for res in body['results']:
            feature_indices.append(int(res['index']))
        return feature_indices[:10]
    else:
        print(f"Failed to fetch concept features: {r.status_code} - {r.reason}", file=sys.stderr)
        print(f"model_id={model_id}, neuronpedia_sae_id={neuronpedia_sae_id}, concept={concept}", file=sys.stderr)
        return []
    


