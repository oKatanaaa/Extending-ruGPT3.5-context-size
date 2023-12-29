CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
    python test_perplexity.py ./long_text_wiki.prk 2048 4096 --n_pos 4096 --adapter_path ./experiments_wiki_4096_4bit/checkpoint-50

#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
#    python test_perplexity.py ./long_text_wiki.prk 2048 4096 6144 --n_pos 6144 --adapter_path ./experiments_wiki_6144

