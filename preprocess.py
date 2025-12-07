from datasets import load_dataset, disable_progress_bar
from transformers import AutoTokenizer
import os
from transformer_lens.loading_from_pretrained import get_official_model_name
from probes import tl_model_names

# prepocess dataset by:
# 1. keeping only the user initial message
# 2. tokenizing with the model tokenizer and chat template
# 3. filter tokenized examples longer than L_max
# 4. save only input_ids to disk

def preprocess(dataset_name: str, tl_model_name: str, L_max: int = 256):
    hf_model_name = get_official_model_name(tl_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    print(f"Loading dataset {dataset_name}", file=os.sys.stderr)
    dataset = load_dataset(dataset_name, "main", split="train", streaming=False)


    # def tokenize_example(example: dict) -> dict:
    #     new_conv = []
    #     new_conv.append(example["conversation"][0])  # keep only the first message
    #     tokens = tokenizer.apply_chat_template(new_conv,
    #                                            tokenize=True,
    #                                            add_generation_prompt=True,
    #                                            padding=False, # no padding 
    #                                            truncation=False, # no truncation 
    #                                            return_tensors=None) # return as list of token ids
    #     return {"input_ids": tokens, "length": len(tokens), "conversation": new_conv}

    def tokenize_question(example: dict) -> dict:
        conv = [
            {
                "role": "user",
                "content": example["question"]
            }
        ]
        tokens = tokenizer.apply_chat_template(conv,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               padding=False, # no padding 
                                               truncation=False, # no truncation 
                                               return_tensors=None) # return as list of token ids
        return {"input_ids": tokens, "length": len(tokens)} 
    
    # print("tokenizing only the initial user message", file=os.sys.stderr)
    # tokenized_dataset = dataset.map(tokenize_example)

    print("tokenizing only the question", file=os.sys.stderr)
    tokenized_dataset = dataset.map(tokenize_question)

    print(f"Filtering examples longer than {L_max} tokens", file=os.sys.stderr)
    filtered_dataset = tokenized_dataset.filter(lambda x: x["length"] <= L_max)


    # # remove all columns except input_ids, length and conversation
    # filtered_dataset = filtered_dataset.remove_columns([col for col in filtered_dataset.column_names if col not in ["input_ids", "length", "conversation"]])

    # save to disk
    path = f"data/preprocessed/{dataset_name}/{tl_model_name}_L={L_max}"
    os.makedirs(path, exist_ok=True)
    filtered_dataset.save_to_disk(path)
    print(f"Preprocessed dataset saved to {path}", file=os.sys.stderr)


if __name__ == "__main__":
    disable_progress_bar()
    for tl_model_name in tl_model_names.values():
        print(f"Preprocessing for model: {tl_model_name}", file=os.sys.stderr)
        preprocess("openai/gsm8k", tl_model_name, L_max=256)