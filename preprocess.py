from datasets import load_dataset, disable_progress_bar
from transformers import AutoTokenizer
import os

# prepocess dataset by:
# 1. keeping only the user initial message
# 2. tokenizing with the model tokenizer and chat template
# 3. filter tokenized examples longer than L_max
# 4. save only input_ids to disk


def preprocess(dataset_name: str, model_name: str, L_max: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", streaming=False)

    def tokenize_example(row: dict) -> dict:
        new_conv = []
        new_conv.append(row["conversation"][0])  # keep only the first message
        tokens = tokenizer.apply_chat_template(
            new_conv,
            tokenize=True,
            add_generation_prompt=True,
            padding=False,  # no padding
            truncation=False,  # no truncation
            return_tensors=None,
        )  # return as list of token ids
        assert isinstance(tokens, list), "tokens is not type of list"
        assert not isinstance(tokens[0], list), "tokens is a nested list"
        return {"input_ids": tokens, "length": len(tokens), "conversation": new_conv}

    print("tokenizing only the initial user message")
    tokenized_dataset = dataset.map(tokenize_example)

    # def tokenize_question(example: dict) -> dict:
    #     conv = [
    #         {
    #             "role": "user",
    #             "content": example["question"]
    #         }
    #     ]
    #     tokens = tokenizer.apply_chat_template(conv,
    #                                            tokenize=True,
    #                                            add_generation_prompt=True,
    #                                            padding=False, # no padding
    #                                            truncation=False, # no truncation
    #                                            return_tensors=None) # return as list of token ids
    #     return {"input_ids": tokens, "length": len(tokens)}

    # print("tokenizing only the question")
    # tokenized_dataset = dataset.map(tokenize_question)

    print(f"Filtering examples longer than {L_max} tokens")
    filtered_dataset = tokenized_dataset.filter(lambda x: x["length"] <= L_max)

    # remove all columns except input_ids, length and conversation
    filtered_dataset = filtered_dataset.remove_columns(
        [
            col
            for col in filtered_dataset.column_names
            if col not in ["input_ids", "length", "conversation"]
        ]
    )

    # save to disk
    path = f"data/preprocessed/{dataset_name}/{model_name}_L={L_max}"
    os.makedirs(path, exist_ok=True)
    filtered_dataset.save_to_disk(path)
    print(f"Preprocessed dataset saved to {path}")


if __name__ == "__main__":
    disable_progress_bar()
    dataset_name = "allenai/WildChat-1M"
    for model_name in [
        "Qwen/Qwen2.5-0.5B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "google/gemma-2-9b-it",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "mistralai/Ministral-8B-Instruct-2410",
    ]:
        print(f"Preprocessing {dataset_name} for model: {model_name}")
        preprocess(dataset_name=dataset_name, model_name=model_name, L_max=256)
