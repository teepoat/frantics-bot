from transformers import GPT2Tokenizer
from pathlib import Path
from .constants import CHECKPOINT_PATH


def modified_tokenizer(model_name="ai-forever/rugpt3small_based_on_gpt2", cache_dir="model_cache", data_path=Path(CHECKPOINT_PATH)):
    if cache_dir:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(data_path / cache_dir))
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    special_tokens_dict = {
        "additional_special_tokens": [
            "<user>", 
            "<says>", 
            "<response>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_tokens(["<laugh>"])
    return tokenizer