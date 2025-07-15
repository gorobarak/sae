import sys




def print_and_write(message, lines):
    print(message, file=sys.stderr)
    lines.append(message + "\n")

def class_idx_to_class_name_dbpedia(class_idx):
    class_names = [
        "company", "educational_institution", "artist", "athlete", 
        "office_holder", "mean_of_transportation", "building", 
        "natural_place", "village", "animal", "plant", 
        "album", "film", "written_work"
    ]
    return class_names[class_idx]

def get_file_prefix(**kwargs):
    prefix = ""
    for key, value in kwargs.items():
        if value:
            prefix += f"{key}_"
    return prefix