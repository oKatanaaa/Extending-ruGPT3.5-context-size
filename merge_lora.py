import typer
from typing_extensions import Annotated
from peft import PeftModel
import os
import torch

from data_utils import prepare_tokenizer_dataset
from model_utils import get_model, upsample_positional_embeddings_inplace
from config import load_training_config


def merge(
    config_path: Annotated[str, typer.Option()] = "training_config.yaml",
    adapter_path: Annotated[str, typer.Option()] = None
):
    config = load_training_config(config_path)
    
    if adapter_path is None:
        adapter_path = config.output_dir
        
    # Do not load in 8-bit to be able to merge
    # Do not load on gpu to avoid OOM
    model = get_model(load_in_8bit=False, device_map=None, dtype=torch.float32)
    upsample_positional_embeddings_inplace(model, new_npos=config.max_ctx_len)
    modeltomerge = PeftModel.from_pretrained(model, adapter_path)
    merged_model = modeltomerge.merge_and_unload()
    merged_model.save_pretrained(os.path.join(adapter_path, "merged"))


if __name__ == "__main__":
    typer.run(merge)