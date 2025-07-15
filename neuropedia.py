import requests
import sys


def get_description(feature_idx, model_id="gemma-2-2b", neuronpedia_id="24-gemmascope-res-16k"):
    r = requests.get(
    f"https://www.neuronpedia.org/api/feature/{model_id}/{neuronpedia_id}/{feature_idx}"
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