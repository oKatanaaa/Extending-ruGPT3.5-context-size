import argparse
from transformers import AutoTokenizer
from peft import PeftModel

from gpt2_patch import replace_gpt2_attn_with_flash_attn
print('Patching gpt2')
replace_gpt2_attn_with_flash_attn()
print('Patched gpt2')
from model_utils import upsample_positional_embeddings_inplace, get_model
from utils import eval_perplexity, load_test_data


def main(dataset_filepath, n_samples, n_pos, n_pos_list, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruGPT-3.5-13B')
    print('Loading the dataset...')
    articles = load_test_data(dataset_filepath, n_samples)
    print('Loaded.')
    
    print('Loading the model...')
    model = get_model(max_ctx_len=n_pos)
    if adapter_path is not None:
        print(f'Received adapter path {adapter_path}. Loading PEFT model...')
        # We need to upsample the size of the positional embeddings.
        # Otherwise weights won't load.
        upsample_positional_embeddings_inplace(model, new_npos=n_pos)
        model = PeftModel.from_pretrained(model, adapter_path)
    print('Loaded.')
    
    if adapter_path is None:
        print('Upsampling positional embeddings...')
        upsample_positional_embeddings_inplace(model, new_npos=n_pos)
        print('Done.')
    
    print('Evaluating perplexity...')
    for ctx_size in n_pos_list:
        perplexity = eval_perplexity(
            articles, tokenizer, 
            model, max_tokens=ctx_size
        )
        print('Ctx size:', ctx_size)
        print('Perplexity:', perplexity)
    print('Done.')


if __name__ == "__main__":
    # Example call `python test_perplexity_no_finetune.py long_text_wiki.prk 2048 4096 8192`
    parser = argparse.ArgumentParser(description="Evaluate the perplexity of a transformer model on a dataset")
    parser.add_argument("dataset_filepath", type=str, help="Path to the dataset file")
    parser.add_argument('n_pos_list', nargs='+', type=int, help="List of ctx sizes to test.")
    parser.add_argument("--n_samples", type=int, help="Number of samples to evaluate", default=100)
    parser.add_argument("--n_pos", type=int, help="Number of positional embeddings", default=2048)
    parser.add_argument("--adapter_path", type=str, help='Path to LoRA adapter.', default=None)

    args = parser.parse_args()
    main(args.dataset_filepath, args.n_samples, args.n_pos, args.n_pos_list, args.adapter_path)
