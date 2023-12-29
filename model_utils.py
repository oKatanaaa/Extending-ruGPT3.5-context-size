from transformers import LlamaForCausalLM, AutoModelForCausalLM
from data_utils import BASE_MODEL
import torch
import torch.nn as nn


def get_model(load_in_8bit=True, dtype=torch.float32, device_map='auto', max_ctx_len=None):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # 8 bit is numerically unstable on V100, use 4bit only
        load_in_4bit=load_in_8bit,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.use_cache = False
    if max_ctx_len is not None:
        model.config.n_positions = max_ctx_len
    return model


def upsample_positional_embeddings_inplace(model, new_npos=4096, mode='linear'):
    original_wpe = model.transformer.wpe
    if original_wpe.weight.shape[0] == new_npos:
        print(f'Received new_npos={new_npos}, but wpe has the same npos.')
        print('Aborting upsample.')
        return
    new_wpe = interpolate_wpe(original_wpe, target_length=new_npos, mode=mode)
    model.transformer.wpe = new_wpe
    
    
def interpolate_wpe(wpe: nn.Embedding, target_length=4096, mode='linear'):
    new_embed = nn.Embedding(num_embeddings=target_length, embedding_dim=wpe.embedding_dim)
    # Upsample original weights
    weight = wpe.weight.data
    if weight.ndim == 2:
        weight = weight.unsqueeze(0)
        weight = weight.transpose(1, 2)
    weight = nn.functional.interpolate(weight, size=(target_length), mode=mode)
    weight = weight.squeeze(0).transpose(0, 1).contiguous()
    new_embed.weight = nn.Parameter(weight)
    return new_embed


def extend(wpe: nn.Embedding, target_length=4096, mode='linear'):
    device = wpe.weight.data.device
    dtype = wpe.weight.data.dtype
    new_embed = nn.Embedding(num_embeddings=target_length, embedding_dim=wpe.embedding_dim, device=device, dtype=dtype)
    # Upsample original weights
    weight = wpe.weight.data.cpu().to(torch.float16).numpy()
    orig_n_ctx = weight.shape[0]
    new_embed.weight.data[:orig_n_ctx] = wpe.weight.data
    new_embed.weight.data[orig_n_ctx:] = torch.randn((target_length - orig_n_ctx, wpe.embedding_dim), dtype=dtype, device=device) * torch.std(wpe.weight.data)
    new_embed.weight.data = new_embed.weight.data.contiguous()
    return new_embed


def interpolate_wpe_v2(wpe: nn.Embedding, target_length=4096, mode='linear'):
    return extend(wpe, target_length, mode)