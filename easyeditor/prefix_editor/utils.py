import torch
import numpy as np
import random
import torch.nn as nn
prefix_len = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb_gptj(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed





# 获取倒数第二层的last token  或  获取倒数第二层的所有query所有token
def register_hook(model,prompts,hparams,query_last_token):
    layer_idx = hparams.n_layer-2
    MODEL_LAYERS = {
        "gpt2":    model.transformer.h[layer_idx] if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.h[layer_idx] if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.h[layer_idx] if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.layers[layer_idx] if "llama-2-7b"   in hparams.M_name else None,
        "llama-3.1-8b": model.model.layers[layer_idx] if "llama-3.1-8b"   in hparams.M_name else None
    }
    
    query_len = prompts.shape[1]
    def hook(module,input,output):
        # print(f"hidden state: {output[0].shape}")
        # print(f"attn: {output[2].shape}")
        hidden_states = output[0].clone().detach()
        last_token = hidden_states[:,query_len-1,:]
        # mean_token = torch.mean(hidden_states[:,query_len-10:query_len,:],dim=1)

        '''
        p = 0.01
        probs = torch.full(last_token.shape, p).to(hparams.device)
        mask = torch.bernoulli(probs)
        token = torch.where(mask==1,last_token,mean_token)
        query_last_token.append(token)
        '''
        # p = 0.9  # GPT2 √
        # p = 0.999
        # tokens = torch.cat((last_token*1,mean_token*0),dim=-1)
        # tokens = torch.cat((last_token*0,mean_token*p),dim=-1)
        query_last_token.append(last_token)
        
        # query_last_token.append(hidden_states[:,query_len-3,:])
        # query_last_token.append(torch.mean(hidden_states[:,query_len-3:query_len,:],dim=1)) 
        # query_last_token.append(torch.mean(hidden_states[:,query_len-5:query_len,:],dim=1))  # GS=91%
        # query_last_token.append(torch.mean(hidden_states[:,query_len-6:query_len,:],dim=1))  # GS=93%
        # query_last_token.append(torch.mean(hidden_states[:,query_len-8:query_len,:],dim=1)) 

    layer = MODEL_LAYERS[hparams.M_name]
    handle = layer.register_forward_hook(hook)

    return handle
    


# 获取倒数第二层的last token  或  获取倒数第二层的所有query所有token
def register_attn_hook(model,prompts,hparams,query_last_token):
    layer_idx = hparams.n_layer-2
    ATTN_LAYERS = {
        "gpt2":    model.transformer.h[layer_idx].attn if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.h[layer_idx].attn if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.h[layer_idx].attn if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.layers[layer_idx].self_attn if "llama-2-7b"   in hparams.M_name else None,
        "llama-3.1-8b": model.model.layers[layer_idx].self_attn if "llama-3.1-8b"   in hparams.M_name else None
    }

    
    query_len = prompts.shape[1]
    def attn_hook(module,input,output):
        hidden_states = output[0].clone().detach()
        last_token = hidden_states[:,query_len-1,:]  # 因为是left padding 所以可以直接取！
        # mean_token = torch.mean(hidden_states[:,query_len-5:query_len,:],dim=1)

        # p = 0.1
        # tokens = torch.cat((last_token*p,mean_token*(1-p)),dim=-1)
        # tokens = torch.cat((last_token*1,mean_token*0),dim=-1)
        query_last_token.append(last_token)
        # print(f"last_token: {last_token.shape}")

    layer = ATTN_LAYERS[hparams.M_name]
    handle = layer.register_forward_hook(attn_hook)

    return handle


def register_midattn_hook(model,prompts,prompt_target_ids,hparams,query_last_token):
    layer_idx = hparams.n_layer-2
    ATTN_LAYERS = {
        "gpt2":    model.transformer.h[layer_idx].attn if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.h[layer_idx].attn if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.h[layer_idx].attn if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.layers[layer_idx].self_attn if "llama-2-7b"   in hparams.M_name else None,
        "llama-3.1-8b": model.model.layers[layer_idx].self_attn if "llama-3.1-8b"   in hparams.M_name else None
    }
    # ATTN_LAYERS = {
    #     "gpt2":    model.transformer.h[layer_idx] if 'gpt2'    in hparams.M_name else None,
    #     "gpt2-xl": model.transformer.h[layer_idx] if 'gpt2-xl' in hparams.M_name else None,
    #     "gpt-j-6b": model.transformer.h[layer_idx] if 'gpt-j-6b' in hparams.M_name else None,
    #     "llama-2-7b": model.model.layers[layer_idx] if "llama-2-7b"   in hparams.M_name else None
    # }
    batchsize,query_len = prompts.shape[0],prompts.shape[1]
    from collections import defaultdict
    def attn_hook(module,input,output):
        hidden_states = output[0].clone().detach()
        last_token = hidden_states[:,query_len-1,:]
        query_last_token.append(last_token)
        
        # max = hidden_states
        # max = nn.functional.softmax(max,dim=-1)
        # max = torch.argmax(max,dim=1)
        
        # logits = model.lm_head(last_token)
        # output = nn.functional.softmax(logits,dim=-1)
        # output = torch.argmax(output,dim=-1)
        # # print(f"logits: {logits}")
        # # print(f"output: {output}")
        # ess = last_token-model.lm_head.weight[output[0]]
        # # print(f"{ess.shape}")
        
        # query_last_token.append(ess)
        
        # print(f"logits: {logits.shape}")
        # print(f"output: {output.shape}")
        # print(f"label: {prompt_target_ids[:,query_len]}")
        # print(f"hidden_states: {hidden_states}")
        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"max: {max.shape}")
        # print(f"max: {max}")
        '''
        max = nn.functional.softmax(hidden_states,dim=-1)
        idxs = torch.argmax(max,dim=1)
        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"idxs: {idxs}")
        # print(f"idxs: {idxs.shape}")
        
        batch_tokens = []
        cnt = [defaultdict(int) for _ in range(batchsize)]
        for bs in range(batchsize):
            argmax_idx = idxs[bs]
            # print(f"argmax_idx: {argmax_idx.shape}")
            for idx in argmax_idx.tolist():
                cnt[bs][idx]+=1
            
            tokens = []
            for k,v in cnt[bs].items():
                if k!=0 and v>=100:
                    tokens.append(hidden_states[bs,k,:])
            # print(f"tokens: {tokens}")
            if len(tokens)!=0:tokens = torch.stack(tokens)
            else: 
                tokens = torch.zeros_like(hidden_states[:,query_len-1,:])  # 存在argmax多数为0的情况
                # print(f"000000"*100)
            mean_token = torch.mean(tokens,dim=0)
            # print(f"mean_token: {mean_token.shape}")
            
            batch_tokens.append(mean_token)
        batch_tokens = torch.stack(batch_tokens)
        # print(f"batch_tokens: {batch_tokens.shape}")
        query_last_token.append(batch_tokens)
        '''
        
        # for idx in idxs:
        #     cnt[idx]+=1
        # token_idxs = []
        # for k,v in cnt.items():
        #     if k!=0 and v>=100:
        #         token_idxs.append(k)
        # print(f"token_idxs: {token_idxs}")
        # print(f"cnt: {sorted(cnt.items(),key=lambda x:(-x[1],x[0]))}")


    layer = ATTN_LAYERS[hparams.M_name]
    handle = layer.register_forward_hook(attn_hook)

    return handle




# register_token  不能是最后一层  因为没有attn了
def register_token_hook(model,prefix_embds,hparams,device):
<<<<<<< HEAD
    layer_idx = hparams.n_layer-30
=======
    layer_idx = hparams.prefix_layer
>>>>>>> f27e33f4703e2872b0b009872582979d797205c3
    MODEL_LAYERS = {
        "gpt2":    model.transformer.h[layer_idx] if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.h[layer_idx] if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.h[layer_idx] if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.layers[layer_idx] if "llama-2-7b"   in hparams.M_name else None,
        "llama-3.1-8b": model.model.layers[layer_idx] if "llama-3.1-8b"   in hparams.M_name else None
    }
    
    # query_len = prompts.shape[1]
    def token_hook(module,input,output):
        hidden_states = output[0] #.clone().detach()
        # print(f"before hidden_states[:,:prefix_len,:]:{hidden_states[:,:prefix_len,:]}")
        # prefix_embds.requires_grad_(True)
        hidden_states[:,:prefix_len,:] += prefix_embds*10
        # hidden_states[:,:prefix_len,:] = prefix_embds
        # hidden_states = torch.cat((prefix_embds,hidden_states),dim=1)
        # hidden_states[:,:prefix_len,:].requires_grad_(True)
        # print(f"after hidden_states[:,:prefix_len,:]:{hidden_states[:,:prefix_len,:]}")
        # print(f"prefix_embds:{prefix_embds}")
        
    layer = MODEL_LAYERS[hparams.M_name]
    handle = layer.register_forward_hook(token_hook)

    return handle



def register_subject_hook(model,prompts,hparams,subject_attn_token,start_end):
    layer_idx = hparams.subject_layer
    # ATTN_LAYERS = {
    #     "gpt2":    model.transformer.h[layer_idx].attn if 'gpt2'    in hparams.M_name else None,
    #     "gpt2-xl": model.transformer.h[layer_idx].attn if 'gpt2-xl' in hparams.M_name else None,
    #     "gpt-j-6b": model.transformer.h[layer_idx].attn if 'gpt-j-6b' in hparams.M_name else None,
    #     "llama-2-7b": model.model.layers[layer_idx].self_attn if "llama-2-7b"   in hparams.M_name else None
    # }
    ATTN_LAYERS = {
        "gpt2":    model.transformer.h[layer_idx] if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.h[layer_idx] if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.h[layer_idx] if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.layers[layer_idx] if "llama-2-7b"   in hparams.M_name else None,
        "llama-3.1-8b": model.model.layers[layer_idx] if "llama-3.1-8b"   in hparams.M_name else None
    }
    
    query_len = prompts.shape[1]
    def attn_hook(module,input,output):
        hidden_states = output[0].clone().detach()
        batchsize = hidden_states.shape[0]
        # last_token = hidden_states[:,query_len-1,:]
        layer_tokens = []
        # print(f"start_end: {start_end}  {len(start_end)}")
        # if len(start_end)==0:
        #     print(f"hidden_states[:,start:end+1,:]: {hidden_states[:,start:end+1,:].shape}")
        #     subject_attn_token.append(torch.zeros_like(hidden_states[:,start:end+1,:]))
        #     return # 仅针对locality
        for i in range(batchsize):
            if len(start_end)>0:
                start = start_end[i][0]
                end = start_end[i][1]
                mean_token = torch.mean(hidden_states[i,start:end+1,:].unsqueeze(dim=0),dim=1)
                layer_tokens.append(mean_token)
            else:
                mean_token = torch.zeros_like(hidden_states[0,0,:].unsqueeze(dim=0))
                layer_tokens.append(mean_token)
            # print(f"mean_token: {mean_token.shape}")
        tokens = torch.cat(layer_tokens,dim=0)
        subject_attn_token.append(tokens)
        # print(f"token: {tokens.shape}")
    
    layer = ATTN_LAYERS[hparams.M_name]
    handle = layer.register_forward_hook(attn_hook)

    return handle


def get_kv(model,prefix_embds,hparams,device):
    layer_idx = hparams.n_layer-2

    if 'gpt2' in hparams.M_name or 'gpt2-xl' in hparams.M_name:
        out = model.transformer.h[layer_idx].attn(prefix_embds,use_cache=True)#.split(768, dim=2)
        key,value = out[1]
        # print(f"====prefix_embds: {prefix_embds.shape}")
        # print(f"key: {key.shape}")
        # print(f"value: {value.shape}")


    elif 'gpt-j-6b' in hparams.M_name:
        position_ids = torch.arange(0, prefix_len, device=device).unsqueeze(0)
        # query = model.transformer.h[layer_idx].attn.q_proj(prefix_embds)
        key = model.transformer.h[layer_idx].attn.k_proj(prefix_embds)
        value = model.transformer.h[layer_idx].attn.v_proj(prefix_embds)

        # query = model.transformer.h[layer_idx].attn._split_heads(query, 16, 256, True)
        key = model.transformer.h[layer_idx].attn._split_heads(key, 16, 256, True)
        value = model.transformer.h[layer_idx].attn._split_heads(value, 16, 256, False)

        embed_positions = create_sinusoidal_positions(2048, 64).to(hparams.device).repeat(position_ids.shape[0], 1, 1)
        # embed_positions = model.transformer.h[layer_idx].attn._get_embed_positions(position_ids)
        # print(f"embed_positions: {embed_positions.shape}")

        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        # print(f"repeated_position_ids: {repeated_position_ids.shape}")
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        rotary_dim = 64
        if rotary_dim is not None:
            k_rot = key[:, :, :, : rotary_dim]
            k_pass = key[:, :, :, rotary_dim :]
            k_rot = apply_rotary_pos_emb_gptj(k_rot, sin, cos)
            key = torch.cat([k_rot, k_pass], dim=-1)

            # q_rot = query[:, :, :, : rotary_dim]
            # q_pass = query[:, :, :, rotary_dim :]
            # q_rot = apply_rotary_pos_emb_gptj(q_rot, sin, cos)
            # query = torch.cat([q_rot, q_pass], dim=-1)

        key = key.permute(0, 2, 1, 3)
        # query = query.permute(0, 2, 1, 3)

    # elif "llama-2-7b" in hparams.M_name:
    elif 'llama' in hparams.M_name:
        position_ids = torch.arange(0, prefix_len, device=device).unsqueeze(0)
        bsz, q_len, _ = prefix_embds.size()
        query = model.model.layers[layer_idx].self_attn.q_proj(prefix_embds)
        key = model.model.layers[layer_idx].self_attn.k_proj(prefix_embds)
        value = model.model.layers[layer_idx].self_attn.v_proj(prefix_embds)   
        query = query.view(bsz, q_len, 32, 128).transpose(1, 2)
        key = key.view(bsz, q_len, 32, 128).transpose(1, 2)
        value = value.view(bsz, q_len, 32, 128).transpose(1, 2)         
        cos, sin = model.model.layers[layer_idx].self_attn.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)  
    
    return key,value



def get_EMBD_LAYER(model,hparams):
    EMBD_LAYER = {
        "gpt2":    model.transformer.wte if 'gpt2'    in hparams.M_name else None,
        "gpt2-xl": model.transformer.wte if 'gpt2-xl' in hparams.M_name else None,
        "gpt-j-6b": model.transformer.wte if 'gpt-j-6b' in hparams.M_name else None,
        "llama-2-7b": model.model.embed_tokens if "llama-2-7b"  in hparams.M_name else None,
        "llama-3.1-8b": model.model.embed_tokens if "llama-3.1-8b"  in hparams.M_name else None
    }
    return EMBD_LAYER



# # # 仅batchszie=1
# def get_start_end(tok,M_name,input_ids,subject_ids):
#     pad_token_id = tok.eos_token_id
#     if 'gpt2' in M_name or 'gpt2-xl' in M_name or "gpt-j-6b" in M_name:
#         subject_ids = subject_ids
#     elif "llama-2-7b" in M_name:
#         subject_ids = subject_ids[:,1:]

#     # print(f"pad_token_id: {pad_token_id}")
#     start_end = []
#     subject_len = subject_ids.shape[1]

#     # print(f"input_ids:{input_ids}")
#     # print(f"input_ids:{input_ids.shape}")
#     # print(f"subject_ids:{subject_ids}")
#     # print(f"subject_ids:{subject_ids.shape}")
#     # # print(f"len(input_ids):{len(input_ids)}")
    
#     # inp_ids_list =[str(id) for id in input_ids[0].cpu().tolist() if id!=tok.pad_token_id]
#     # subj_ids_list =[str(id) for id in subject_ids[0].cpu().tolist() if id!=tok.pad_token_id]
#     # print(f"inp_ids_list: {inp_ids_list}")
#     # print(f"subj_ids_list: {subj_ids_list}")
#     for i in range(len(input_ids)):  # 遍历A中每个ids串
#         # print(f"i:{i}")
#         # print(f"input_ids[i]:{input_ids[i]}")
#         INSERTED = False
#         id_seq = input_ids[i]
#         target_seq = subject_ids[i]
#         if 'gpt2' in M_name or 'gpt2-xl' in M_name or "gpt-j-6b" in M_name:
#             pad_mask = torch.where(target_seq==pad_token_id,1,0)  # 用于计算 非<PAD>个数  并删去<PAD>
#             pad_len = sum(pad_mask)
#             target_seq = subject_ids[i][pad_len:]
#             target_len = len(target_seq)
#             # print(f"target_seq: {target_seq}   pad_len:{pad_len}  target_len:{target_len}")
#             # print(f"id_seq: {id_seq}   ")
#         elif "llama-2-7b" in M_name:
#             bos_mask = torch.where(target_seq==1,1,0)  # 用于计算 非<BOS><EOS>个数  并删去<BOS><EOS>
#             eos_mask = torch.where(target_seq==2,1,0)  # 用于计算 非<BOS><EOS>个数  并删去<BOS><EOS>
#             bos_len = sum(bos_mask)
#             eos_len = sum(eos_mask)
#             target_seq = subject_ids[i][bos_len+eos_len:-1]
#             target_len = len(target_seq)

#         for j in range(len(id_seq) - target_len + 1):  # 遍历可能的起始位置
#             if torch.equal(id_seq[j:j + target_len], target_seq):  # 比较切片是否和目标字串相等
#                 # start_end.append([j,j+target_len-1])
#                 start_end.append([j,j+subject_len-1])
#                 INSERTED = True
#                 # start_end.append([j+prefix_len,j+subject_len-1+prefix_len])
#                 break  # 找到就跳出内层循环，进入下一个ids串与对应字串的匹配
#         if INSERTED==True:continue
#         for j in range(len(id_seq) - (target_len-1) + 1):  # 遍历可能的起始位置
#             # print(f"id_seq: {id_seq[j:j + (target_len-1)]}   target_seq[1:]:{target_seq[1:]}")
            
#             if torch.equal(id_seq[j:j + (target_len-1)], target_seq[1:]):  
#                 start_end.append([j-1,j+(target_len-1)-1])
#                 INSERTED = True
#                 break  
#             # if j==len(id_seq) - target_len + 1-1:
#         if INSERTED==True:continue
#         for j in range(len(id_seq) - (target_len-1) + 1):  # 遍历可能的起始位置
#             # print(f"id_seq: {id_seq[j:j + (target_len-1)]}   target_seq[1:]:{target_seq[:-1]}")
#             if torch.equal(id_seq[j:j + (target_len-1)], target_seq[:-1]): 
#                 start_end.append([j,j+(target_len-1)-1+1])
#                 INSERTED = True
#                 break  
#         if INSERTED==True:continue
#         for j in range(len(id_seq) - (target_len-2) + 1):  # 遍历可能的起始位置
#             # print(f"id_seq: {id_seq[j:j + (target_len-1)]}   target_seq[1:]:{target_seq[:-1]}")
#             if torch.equal(id_seq[j:j + (target_len-2)], target_seq[1:-1]): 
#                 start_end.append([j-1,j+(target_len-2)-1+1])
#                 INSERTED = True
#                 break  

#     return start_end

def get_start_end(prompt_ids,trained_fusion_ids):
    prompt_ids = prompt_ids[0].cpu()   
    # subject_ids = subject_ids[0].cpu()
    start_end = []
    start_idx,end_idx = -1,-1
    max_len = 0

    FIND = False
    for start in range(prompt_ids.shape[0]):
        key = ""
        for end in range(start, prompt_ids.shape[0]):
            key = key + str(prompt_ids[end].item()) +  '_'
            # print(f"#key: {key}") 
            patterns = ['x_match','match_x','x_x_match',"match_x_x"]    # 空间换时间
            if key[:-1] in trained_fusion_ids['match']:
                max_len = end-start+1
                start_idx,end_idx = start,end
                FIND = True
                continue
            if end-start+1>max_len:
                for pattern in patterns:
                    if key[:-1] in trained_fusion_ids[pattern]:
                        max_len = end-start+1
                        start_idx,end_idx = start,end
                        break
        if FIND:break
    if start_idx!=-1 and end_idx!=-1:start_end = [[start_idx,end_idx]]
    # print(f"start_end: {start_end}")
    return start_end

import re
# 仅适用于batchsize=1
# 将输入的句子进行 暴力检索；若trained_fusion_ids存在这个subject_ids，则更新这个subject（方法不好，但能用）
def find_subject(prompt_ids, trained_fusion_ids):
    prompt_ids = prompt_ids[0].cpu()   
    start_end = []
    start_idx,end_idx = -1,-1
    subject_ids = None
    max_len = 0
    FIND = False
    for start in range(prompt_ids.shape[0]):
        key = ""
        for end in range(start, prompt_ids.shape[0]):
            key = key + str(prompt_ids[end].item()) +  '_'
            # print(f"#key: {key}") 
            patterns = ['x_match','match_x','x_x_match',"match_x_x"]
            
            if key[:-1] in trained_fusion_ids['match']:
                subject_ids = prompt_ids[start:end+1].unsqueeze(dim=0)
                max_len = end-start+1
                start_idx,end_idx = start,end
                FIND = True
                continue
            if end-start+1>max_len:
                for pattern in patterns:
                    if key[:-1] in trained_fusion_ids[pattern]:
                        if pattern=='x_match':subject_ids = prompt_ids[(start-1):end+1].unsqueeze(dim=0)
                        elif pattern=='match_x':subject_ids = prompt_ids[start:(end+1)+1].unsqueeze(dim=0)
                        elif pattern=='x_x_match':subject_ids = prompt_ids[(start-2):end+1].unsqueeze(dim=0)
                        elif pattern=='match_x_x':subject_ids = prompt_ids[start:(end+2)+1].unsqueeze(dim=0)
                        max_len = end-start+1
                        start_idx,end_idx = start,end
                        break
        if FIND:break
        #     if key[:-1] in trained_fusion_ids: 
        #         if end-start+1>max_len:
        #             subject_ids = prompt_ids[start:end+1].unsqueeze(dim=0)
        #             max_len = end-start+1
        #             start_idx,end_idx = start,end
        #             FIND = True
        #             continue
        #     if end-start+1>max_len:
        #         for k,_ in trained_fusion_ids.items():
        #             pattern1 = rf'^{key[:-1]}_[0-9]+$'
        #             pattern2 = rf'^{key[:-1]}_[0-9]+_[0-9]+$'
        #             pattern3 = rf'^[0-9]+_{key[:-1]}$'
        #             pattern4 = rf'^[0-9]+_[0-9]+_{key[:-1]}$'
        #             if re.match(pattern1,k):
        #                 subject_ids = prompt_ids[start:(end+1)+1].unsqueeze(dim=0)
        #                 max_len = end-start+1
        #                 start_idx,end_idx = start,end
        #                 break
        #             elif re.match(pattern2,k):
        #                 subject_ids = prompt_ids[start:(end+2)+1].unsqueeze(dim=0)
        #                 max_len = end-start+1
        #                 start_idx,end_idx = start,end 
        #                 break 
        #             elif re.match(pattern3,k):
        #                 subject_ids = prompt_ids[(start-1):end+1].unsqueeze(dim=0)
        #                 max_len = end-start+1
        #                 start_idx,end_idx = start,end
        #                 break
        #             elif re.match(pattern4,k):
        #                 subject_ids = prompt_ids[(start-2):end+1].unsqueeze(dim=0)
        #                 max_len = end-start+1
        #                 start_idx,end_idx = start,end
        #                 break
        #     if FIND:break
        # if FIND:break
    if start_idx!=-1 and end_idx!=-1:start_end = [[start_idx,end_idx]]
    # print(f"trained_fusion_ids: {list(trained_fusion_ids.items())[-1]}")
    # print(f"trained_fusion_ids: {trained_fusion_ids['match']}")
    print(f"subject_ids: {subject_ids}")
    return subject_ids,start_end



def cal_acc(all_metrics,num):
    ES_list = [data['post']['rewrite_acc'][0]  for data in all_metrics[:num]]
    ES = sum(ES_list)/len(ES_list)*100
    print(f"ES: {ES}%")

    GS_list = [data['post']['portability']['Generality_acc'][0]  for data in all_metrics[:num]]
    GS = sum(GS_list)/len(GS_list)*100
    print(f"GS: {GS}%")

    LS_list = [data['post']['locality']['Locality_acc'][0]  for data in all_metrics[:num]]
    LS = sum(LS_list)/len(LS_list)*100
    print(f"LS: {LS}%")

    return


