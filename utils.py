import sys

# Global variables
DBPEDIA_CLASS_NAMES=[
    "Company",
    "Educational institution",
    "Artist",
    "Athlete",
    "Office holder",
    "Mean of transportation",
    "Building",
    "Natural place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written work",
]
WANDB_PROJECT = "dbpedia_concept_ranking"

def print_and_write(message, lines):
    print(message, file=sys.stderr)
    lines.append(message + "\n")

def class_idx_to_class_name_dbpedia(class_idx):
    class_names = [
        "company", "educational institution", "artist", "athlete", 
        "office holder", "mean of transportation", "building", 
        "natural place", "village", "animal", "plant", 
        "album", "film", "written work"
    ]
    return class_names[class_idx]

def get_file_prefix(**kwargs):
    prefix = ""
    for key, value in kwargs.items():
        if type(value) is bool:
            if value:
                prefix += f"{key}_"
        if type(value) is int or type(value) is str:
            prefix += f"{key}={value}_"
    return prefix

def get_file_suffix(**kwargs):
    suffix = ""
    for key, value in kwargs.items():
        if type(value) is bool:
            if value:
                suffix += f"_{key}"
        if type(value) is int or type(value) is str:
            suffix += f"_{key}={value}"
    return suffix