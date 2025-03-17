import json
from fuzzywuzzy import process

api_tokens = {}

def register_token(model_name: str, token: str, base_url: str, override: bool = False):
    """Register a new API token."""
    if not override:
        assert (
            model_name not in api_tokens
        ), f"{model_name} has already been registered."

    api_tokens[model_name] = {"api_key": token, "base_url": base_url}

def get_token(query: str, use_fuzzy: bool = False, threshold: int = 80) -> str:
    """Get an API token."""
    if not use_fuzzy:
        return api_tokens.get(query), query

    matched_model = process.extractOne(query, api_tokens.keys(), score_cutoff=threshold)
    if matched_model:
        return api_tokens[matched_model[0]], matched_model[0]
    
    return None

def load_tokens_from_file(tokens_file: str):
    """Load API tokens from a JSON file."""
    with open(tokens_file, 'r') as file:
        tokens = json.load(file)
        
        for model in tokens:
            # breakpoint()
            register_token(model, tokens[model].get("api_key"), tokens[model].get("base_url"), override=True)

# Example usage:
# Load tokens from a JSON file
load_tokens_from_file('API_tokens.json')
