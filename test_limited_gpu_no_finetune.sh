CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
    python test_perplexity.py ./long_text_wiki.prk 2048 --n_pos 2048

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
    python test_perplexity.py ./long_text_wiki.prk 2048 4096 --n_pos 4096

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
    python test_perplexity.py ./long_text_wiki.prk 2048 4096 6144 --n_pos 6144

