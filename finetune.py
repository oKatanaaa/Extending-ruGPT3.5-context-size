# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from gpt2_patch import replace_gpt2_attn_with_flash_attn, upcast_layer_for_flash_attention

print('Patching gpt2')
replace_gpt2_attn_with_flash_attn()
print('Patched gpt2')

import torch
torch.manual_seed(0)
import random
random.seed(0)

import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType
)
from peft.tuners.lora import LoraLayer


from data_utils import prepare_tokenizer_dataset
from model_utils import get_model, upsample_positional_embeddings_inplace
from config import load_training_config

def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model

def main(
    config_path: Annotated[str, typer.Option()] = "training_config.yaml",
    format_version: Annotated[int, typer.Option()] = 2
):
    config = load_training_config(config_path)
    print(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        # --- Load datasets and model        
        task1 = progress.add_task(description="Loading tokenizer and datasets...", total=None)
        train_data, val_data, tokenizer = prepare_tokenizer_dataset(format_v=format_version, max_ctx_len=config.max_ctx_len, dataset_filepath=config.dataset_filepath)
        progress.advance(task1)
        
        task2 = progress.add_task(description="Loading model...", total=None)
        model = get_model(max_ctx_len=config.max_ctx_len)
        progress.advance(task2)
        
        task2_5 = progress.add_task(description="Upsampling wpe...", total=None)
        upsample_positional_embeddings_inplace(model, new_npos=config.max_ctx_len)
        progress.advance(task2_5)
        
        # --- Setup all training stuff
        task3 = progress.add_task(description="Setting up training...", total=None)
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # IMPORTANT! Unfreeze embedding layer to allow for new token finetuning
            # See https://github.com/huggingface/peft/issues/349#issuecomment-1527059611
            # See https://github.com/huggingface/peft/issues/334
            # See https://github.com/huggingface/peft/pull/337#issuecomment-1527412343
            # Also unfreeze classification head to allow for new token classes
            modules_to_save=config.modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        #upcast_layer_for_flash_attention(model, torch.bfloat16)
        model.print_trainable_parameters()
        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.train_steps,
            learning_rate=config.learning_rate,
            fp16=False,
            logging_steps=config.logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit,
            report_to="tensorboard"
        )
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        progress.advance(task3)
    
    # --- Start training
    print("Start training")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    #with torch.autocast("cuda"):
    trainer.train()
    print('Finished training')
    # --- Save
    model.save_pretrained(config.output_dir)


if __name__ == "__main__":
    typer.run(main)