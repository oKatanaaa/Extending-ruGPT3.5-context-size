CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2 \
    python finetune.py --config-path training_config.yaml --format-version 2