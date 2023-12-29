from datasets import load_dataset
import datasets
from transformers import AddedToken, AutoTokenizer
import numpy as np
import pandas as pd


def get_datagens(n_train_samples=1000, n_test_samples=100, dataset_filepath=None):
    print('Loading dataset...')
    if dataset_filepath is not None:
        print(f'Received dataset_filepath={dataset_filepath}.')
        df = pd.read_parquet(dataset_filepath)
    else:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ru")
        print('Loaded.')
        
        print('Convert to pandas...')
        df = dataset['train'].to_pandas()
        print('Converted.')
    
    print('Splitting...')
    train_df = df.iloc[:n_train_samples]
    test_df = df.iloc[-n_test_samples:]
    print(f'Splitted. Train samples: {len(train_df)}. Test samples: {len(test_df)}')
    
    return Generator(train_df), Generator(test_df)


class Generator:
    def __init__(self, df):
        self.df = df
    
    def __call__(self):
        sample_indices = np.arange(len(self.df))
        np.random.shuffle(sample_indices)
        for ind in sample_indices:
            if len(self.df.iloc[ind].text) < 5:
                continue
            yield {'text': self.df.iloc[ind].text}
    
    def __len__(self):
        return len(self.df)


ACCESS_TOKEN = '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'
BASE_MODEL = "ai-forever/ruGPT-3.5-13B"
MAX_LENGTH = 2048
DEBUG = False


def prepare_tokenizer_dataset(tokenizer_only=False, format_v=2, max_ctx_len=2048, debug=False, dataset_filepath=None):
    global MAX_LENGTH
    global DEBUG
    DEBUG = debug
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    MAX_LENGTH = max_ctx_len
    # Never do tokenizer.pad_token = tokenizer.eos_token as it may break things
    if tokenizer_only:
        return tokenizer
    
    train_gen, test_gen = get_datagens(dataset_filepath=dataset_filepath)
    dataset = datasets.IterableDataset.from_generator(train_gen)
    valdataset = datasets.Dataset.from_generator(test_gen)
    train_data = (
        dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    )
    val_data = (
        valdataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer), load_from_cache_file=False)
    )
    return train_data, val_data, tokenizer


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    if DEBUG:
        result["prompt"] = prompt
    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    tokenized_full_prompt = tokenize(data_point['text'], tokenizer)
    return tokenized_full_prompt