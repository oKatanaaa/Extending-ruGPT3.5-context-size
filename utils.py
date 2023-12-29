import torch
import pandas as pd
from tqdm import tqdm
import numpy as np


def split_list(input_list, sublist_length):
    sublists = []
    for i in range(0, len(input_list), sublist_length):
        sublist = input_list[i:i + sublist_length]
        sublists.append(sublist)
    return sublists


def eval_perplexity(text_list, tokenizer, model, max_tokens=2048):
    text_batches = split_list(text_list, 1)
    input_batches = map(
        lambda text_batch: tokenizer(
            text_batch, 
            return_tensors="pt", 
            max_length=max_tokens, 
            truncation=True
        ),
        text_batches
    )
    
    total_perplexity = 0.0
    n_nans = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(input_batches, total=len(text_batches))):
            outputs = model(**batch, labels=batch["input_ids"])
            nll = outputs.loss.float().cpu().numpy()
            if np.isnan(nll):
                print(f'Encountered nan in batch {i}')
                n_nans += 1
                continue
            # nll is a scalar representing mean nll over labels
            batch_perplexity = np.exp(nll)
            total_perplexity += batch_perplexity
            
    return total_perplexity / (len(text_batches) - n_nans)


def load_test_data(dataset_filepath, n_samples=50):
    df = pd.read_parquet(dataset_filepath)
    articles = df.text.tolist()[-n_samples:]
    return articles


