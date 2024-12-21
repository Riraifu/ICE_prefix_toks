from transformers import GPT2TokenizerFast, GPT2Tokenizer,GPT2LMHeadModel
import torch
from torch import nn
from transformers import GPT2Attention

# class GPT2Attn(GPT2Attention):

#     # def origin_last_attn(self, query, key, value, attention_mask=None, head_mask=None):
#     def origin_last_attn(self, query, key, value, attention_mask=None, head_mask=None):
#         attn_weights = torch.matmul(query, key.transpose(-1, -2))

#         if self.scale_attn_weights:
#             attn_weights = attn_weights / torch.full(
#                 [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
#             )

#         # Layer-wise attention scaling
#         if self.scale_attn_by_inverse_layer_idx:
#             attn_weights = attn_weights / float(self.layer_idx + 1)

#         if not self.is_cross_attention:
#             # if only "normal" attention layer implements causal mask
#             query_length, key_length = query.size(-2), key.size(-2)
#             causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
#             mask_value = torch.finfo(attn_weights.dtype).min
#             # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#             # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#             mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
#             attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)




#         if attention_mask is not None:
#             # Apply the attention mask
#             attn_weights = attn_weights + attention_mask

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
#         attn_weights = attn_weights.type(value.dtype)
#         attn_weights = self.attn_dropout(attn_weights)       #me 还有dropout！！

#         # Mask heads if we want to
#         if head_mask is not None:
#             attn_weights = attn_weights * head_mask      

#         attn_output = torch.matmul(attn_weights, value)     

#         return attn_output, attn_weights


#     # 若[start_idx,end_idx]不存在，则不需要修改causal mask
#     # def modify_last_attn(self, query, key, value, attention_mask=None, head_mask=None):
#     def _attn(self, query, key, value, attention_mask=None, head_mask=None):
#         attn_weights = torch.matmul(query, key.transpose(-1, -2))

#         if self.scale_attn_weights:
#             attn_weights = attn_weights / torch.full(
#                 [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
#             )

#         # Layer-wise attention scaling
#         if self.scale_attn_by_inverse_layer_idx:
#             attn_weights = attn_weights / float(self.layer_idx + 1)

#         if not self.is_cross_attention:
#             # if only "normal" attention layer implements causal mask
#             query_length, key_length = query.size(-2), key.size(-2)
#             causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
#             mask_value = torch.finfo(attn_weights.dtype).min
#             # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#             # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#             mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
#             attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
#             print(f"attn_weights: {attn_weights.shape}")
#             print(f"causal_mask: {causal_mask.shape}")
#             print(f"causal_mask: {causal_mask}")     # causal_mask会影响softmax的结果！！！  所以在softmax前使用和softmax后使用结果是不同的！
#             print('-=-=-=-=--=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-')
#             # print(f"attention_mask: {attention_mask}")
#             # print(f"attention_mask: {attention_mask.shape}")
#             # print(f"attn_weights: {attn_weights.shape}")
#             # 1、更改最后一层attn中的causal_mask即可！
#             # 2、需要对1个batch中的不同样本，单独处理！
#             # 3、如何只修改最后一层？  hook？

#             # 情况1：无padding
#             # 情况2：有padding

#             # 需要传入batch中所有样本的[start_idx,end_idx]
#             exit()


#         if attention_mask is not None:
#             # Apply the attention mask
#             attn_weights = attn_weights + attention_mask

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
#         attn_weights = attn_weights.type(value.dtype)
#         attn_weights = self.attn_dropout(attn_weights)       #me 还有dropout！！

#         # Mask heads if we want to
#         if head_mask is not None:
#             attn_weights = attn_weights * head_mask      

#         attn_output = torch.matmul(attn_weights, value)     

#         return attn_output, attn_weights




# class ModifiedGPT2(GPT2LMHeadModel):
#     def __init__(self, config):
#         super().__init__(config)
#         block_num = len(self.transformer.h)
#         self.transformer.h[block_num-1].attn = GPT2Attn



# # # 自定义的Self-Attention层
# # class MyCustomSelfAttention(torch.nn.Module):
# #     def __init__(self, original_attn):
# #         super().__init__()
# #         self.original_attn = original_attn
# #         # 在这里实现你自己的逻辑
    
# #     def forward(self, hidden_states, *args, **kwargs):
# #         # 使用自定义的逻辑处理hidden_states
# #         return self.original_attn(hidden_states, *args, **kwargs)
