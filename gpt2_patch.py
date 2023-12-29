from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import transformers

from einops import rearrange

from xformers import ops as xops


def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        bsz, q_len, _ = hidden_states.size()

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        #TODO Should we support?
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        assert use_cache is False, "Use cache is not supported"
        present = None
        # if use_cache is True:
        #     present = (key, value)
        # else:
        #     present = None

        assert self.reorder_and_upcast_attn is False, "reorder_and_upcast_attn is not supported yet"
        source_dtype = query.dtype
        
        # [bsz, heads, seq_len, hiddens_per_head]
        q, k, v = map(lambda x: x.transpose(1, 2), [query, key, value])
        #q, k, v = map(lambda x: x.to(torch.float16), [q, k, v])
        # [bsz, seq_len, heads, hiddens_per_head]
        output = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask(),
            #p=self.attn_dropout.p
        )

        #output = output.to(source_dtype)
        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        # else:
        #     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        output = rearrange(output, 'b s h d -> b h s d')
        attn_output = self._merge_heads(output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        
        assert output_attentions is False, "output attentions is not supported yet"
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        #     module.to(torch_dtype)
        if 'wpe' in name or 'ln_1' in name or 'ln_2' in name or 'ln_f' in name:
            module.to(torch_dtype)
        if 'wte' in name:
            module.to(torch_dtype)
    return model


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask



def replace_gpt2_attn_with_flash_attn():
    # transformers.models.gpt2.modeling_gpt2.LlamaModel._prepare_decoder_attention_mask = (
    #     _prepare_decoder_attention_mask
    # )
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward