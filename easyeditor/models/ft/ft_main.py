from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams
# from .visual_llm import show_logits


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    trained_fusion_ids = {},
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    # if copy:
    #     model = deepcopy(model)

    # deltas = execute_ft(model, tok, requests, hparams)
    last_token_hd,subject_attn_token,embd_list,deltas = execute_ft(model, tok, requests, hparams,trained_fusion_ids)

    # with torch.no_grad():
    #     for w_name, upd_matrix in deltas.items():
    #         w = nethook.get_parameter(model, w_name)
    #         if return_orig_weights and w_name not in weights_copy:
    #             weights_copy[w_name] = w.detach().clone()

    #         w[...] += upd_matrix

    # print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    # return model, weights_copy
    return last_token_hd,subject_attn_token,embd_list,model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    trained_fusion_ids={},   
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    # print(f"requests: {requests}")
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    context_provided = 'context' in requests[0] and requests[0]['context'] is not None  # True me
    if context_provided:
        knowledge = [r["context"] for r in requests]  # 每个知识5条context，5N
        context_num = len(knowledge[0])               # 5
        context_num = 1  # me
        
    else:
        knowledge = [[r["prompt"] + r["target_new"] + '. '] for r in requests]   # 每个知识1条context(跟query target一模一样)
        context_num = 1
    
    from ...prefix_editor.utils  import setup_seed
    # Configure optimizer / gradients
    batchsize = len(requests)
    prefix_len = 1
    embd_list = None
    # embd = None
    # embd = nn.Embedding(prefix_len,hparams.n_embd)
    embd_list = [nn.Embedding(prefix_len,hparams.n_embd) for _ in range(batchsize)]
    seeds = []
    for k in range(batchsize):
        embd_list[k] = embd_list[k].to(device)
        seed = random.randint(1,999999)
        seeds.append(seed)
        setup_seed(seed)
        embd_list[k].weight.data.uniform_(-0.1, 0.1)   # # 这个初始化效果特别好！！！
        # nn.init.xavier_uniform_(embd_list[k].weight)
        # nn.init.kaiming_uniform_(embd_list[k].weight,mode='fan_in', nonlinearity='relu')
    dropout = nn.Dropout(p=0.1)


    # # print(f"before embd: {embd.weight}")
    # print(f"last_sc_PAD[0].squeeze().shape: {last_sc_PAD[0].squeeze().unsqueeze(dim=0).shape}")
    # embd.weight.data.copy_( torch.randn(1,768).uniform_(-0.1, 0.1).to(device) + 0.01*last_sc_PAD[0].squeeze().unsqueeze(dim=0))
    # print(torch.norm(last_sc_PAD[0]))
    # embd.weight.data.copy_(torch.norm(last_sc_PAD[0].squeeze().unsqueeze(dim=0)))
    # exit()
    # embd.weight.data = torch.norm(last_sc_PAD[0].squeeze())
    # print(f"after embd: {embd.weight}")

    opt = torch.optim.Adam(
        # [v for _, v in embd.named_parameters()],
        [v  for k in range(batchsize) for _, v in embd_list[k].named_parameters()] +  [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=0.0001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.65)

    for name, w in model.named_parameters():
        w.requires_grad = name in weights   # 需要更新的层 requires_grad=True   冻结的层requires_grad=False
        # print(name,w.requires_grad)
    for k in range(batchsize):
        for name, w in embd_list[k].named_parameters():
            w.requires_grad = True   
            # print(name,w.requires_grad)
    
    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    last_token_hd = []



    # 在这里设置Flag  初始化embd的值为<PAD>的值
    from ...prefix_editor.utils  import get_EMBD_LAYER
    PAD_ID = torch.tensor(tok.pad_token_id).repeat(batchsize,prefix_len).to(device)  # prefix多长  PAD就多长？
    EMBD_LAYER = get_EMBD_LAYER(model,hparams)
    PAD_embds = EMBD_LAYER[hparams.M_name](PAD_ID)

    # last_sc_PAD = []
    # def hook2(module,input,output):
    #     hidden_states = output[0].detach().clone()
    #     last_sc_PAD.append(hidden_states[:prefix_len])
    # if 'gpt2' in hparams.M_name or 'gpt2-xl' in hparams.M_name or 'gpt-j-6b' in hparams.M_name:
    #     PAD_handle = model.transformer.h[hparams.n_layer-1].register_forward_hook(hook2)
    # elif "llama-2-7b" in hparams.M_name:
    #     PAD_handle = model.model.layers[hparams.n_layer-1].register_forward_hook(hook2)
    # # '''↑↑↑me'''
    # PAD_past_kv = model(inputs_embeds=PAD_embds).past_key_values
    # PAD_handle.remove()



    for it in range(hparams.num_steps):
        loss_meter.reset()

        for id, (txt, tgt, ctx) in enumerate(zip(
            chunks(texts, hparams.batch_size), 
            chunks(targets, hparams.batch_size), 
            chunks(knowledge, hparams.batch_size))):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )

            if hparams.objective_optimization in [ 'target_with_context','target_and_completion_with_context']:
                # Prepare tokens for targets, prompts+targets, and contexts+prompts+targets
                tgt = [t for t in tgt for _ in range(context_num)]  # context_num=1
                txt = [t for t in txt for _ in range(context_num)]
                prompts_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                
                # Tokenize the above
                # prompts_targets = tok(prompts_targets, return_tensors="pt", padding='max_length', max_length=35).to(device)
                prompts_targets = tok(prompts_targets, return_tensors="pt", padding=True).to(device)
                prompts_targets_ids = prompts_targets['input_ids']
                
                target_ids = tok(tgt, return_tensors="pt", padding=True).to(device)['input_ids']
                prompts_ids = tok(txt, return_tensors="pt",padding=True).to(device)['input_ids']
                batchsize = prompts_targets['input_ids'].shape[0]
                PAD = torch.tensor(tok.pad_token_id).to(device).repeat(batchsize,prefix_len)  # 
                prompts_ids = torch.cat((PAD,prompts_ids),dim=1)
                prompts_targets_ids = torch.cat((PAD,prompts_targets_ids),dim=1)
                
                # Prepare label mask for targets
                prompt_target_len = prompts_targets_ids.size(1)
                num_tgt_tokens = [int((i != tok.pad_token_id).sum()) for i in target_ids.cpu()]
                label_mask = torch.tensor([[False] * (prompt_target_len - tgt_length)  + [True] * tgt_length for tgt_length in num_tgt_tokens]).to(device)
                if hparams.sample_with_context==False:
                    prefix = prompts_targets
                    batchsize = prompts_targets_ids.shape[0]
                    from ...prefix_editor.utils import get_start_end
                    
                    # 仅支持batchsize=1
                    if 'gpt2' in hparams.M_name or 'gpt2-xl' in hparams.M_name or 'gpt-j-6b' in hparams.M_name:
                        for k in range(batchsize):
                            for subj in [requests[k]['subject'],' '+requests[k]['subject'] ]:
                                subject = [subj]
                                subject_ids = tok(subject, return_tensors="pt", padding=True)['input_ids'].to(device)
                                subject_ids_list = [str(id) for id in subject_ids[0].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
                                if len(subject_ids_list)<=3:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1])]
                                elif len(subject_ids_list)>=4:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1]),('x_x_match',subject_ids_list[2:]),('match_x_x',subject_ids_list[:-2])]
                                for pattern,subj_ids_list in patterns:
                                    trained_fusion_ids[pattern].add('_'.join(subj_ids_list))
                        start_end = get_start_end(prompts_targets_ids,trained_fusion_ids)
                        
                    elif 'llama-2-7b' in hparams.M_name:
                        for k in range(batchsize):
                            for subj in [' '+requests[k]['subject'],requests[k]['subject'],]:
                                subject = [subj]
                                subject_ids = tok(subject, return_tensors="pt", padding=True)['input_ids'].to(device)
                                subject_ids_list = [str(id) for id in subject_ids[0][1:].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
                                if len(subject_ids_list)<=3:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1])]
                                elif len(subject_ids_list)>=4:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1]),('x_x_match',subject_ids_list[2:]),('match_x_x',subject_ids_list[:-2])]
                                for pattern,subj_ids_list in patterns:
                                    trained_fusion_ids[pattern].add('_'.join(subj_ids_list))
                        start_end = get_start_end(prompts_targets_ids,trained_fusion_ids)


                        # _subject = [requests[k]['subject'] for k in range(batchsize)]
                        # _subject_ids = tok(_subject, return_tensors="pt", padding=True)['input_ids'].to(device)
                        # subject_ids_list = [str(id) for id in _subject_ids[0][1:].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
                        # subject_ids_str = '_'.join(subject_ids_list)
                        # trained_fusion_ids[subject_ids_str]=_subject[0]
                        # print(f"#_subject_ids: {_subject_ids}")
                        # # print(f"#subject_ids_list: {subject_ids_list}")
                        # # print(f"#subject_ids_str: {subject_ids_str}")
                        # # print(f"#trained_fusion_ids: {trained_fusion_ids}")
                        # start_end = get_start_end(tok,hparams.M_name,prompts_targets_ids,_subject_ids)


                    elif 'llama-3.1-8b' in hparams.M_name:
                        for k in range(batchsize):
                            for subj in [requests[k]['subject'],' '+requests[k]['subject'] ]:
                                subject = [subj]
                                subject_ids = tok(subject, return_tensors="pt", padding=True)['input_ids'].to(device)
                                subject_ids_list = [str(id) for id in subject_ids[0][1:].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
                                if len(subject_ids_list)<=3:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1])]
                                elif len(subject_ids_list)>=4:
                                    patterns = [('match',subject_ids_list),('x_match',subject_ids_list[1:]),('match_x',subject_ids_list[:-1]),('x_x_match',subject_ids_list[2:]),('match_x_x',subject_ids_list[:-2])]
                                for pattern,subj_ids_list in patterns:
                                    trained_fusion_ids[pattern].add('_'.join(subj_ids_list))
                        start_end = get_start_end(prompts_targets_ids,trained_fusion_ids)


            else:
                print(f"{hparams.objective_optimization} has not been supported yet.")
                raise NotImplementedError

            opt.zero_grad()
            batchsize = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():pass
            else:
                if hparams.objective_optimization == "target_and_completion_with_context":
                    if hparams.static_target and it%hparams.target_update_interval == 0:
                        if it == 0:
                            generated_sequences_list = [
                                model.generate(
                                    **prefix,
                                    do_sample=True,
                                    max_new_tokens=hparams.max_new_tokens,
                                    min_new_tokens=hparams.max_new_tokens,
                                    num_return_sequences=hparams.num_return_sequences,
                                    temperature=hparams.temperature,
                                    output_scores=True,
                                    return_dict_in_generate=True,

                                )
                                for _ in range(hparams.num_steps)
                            ]
                        generated_sequences= generated_sequences_list[it]['sequences']
                    # sequences = torch.cat((torch.repeat_interleave(prompts_targets_ids, hparams.num_return_sequences, dim=0), completions), dim=1)  # num_return_sequences=1
                    # sequences = prompts_targets_ids

                    

                    
                    # prefix = torch.arange(start=0,end=prefix_len).repeat(5, 1).to(device)
                    prefix = torch.arange(start=0,end=prefix_len).repeat(1, 1).to(device)
                    prefix_list = []
                    batchsize = prompts_ids.shape[0]
                    for k in range(batchsize):
                        embd_out = embd_list[k](prefix)
                        embd_out = dropout(embd_out)
                        prefix_list.append(embd_out)
                    prefix_embds = torch.cat(prefix_list,dim=0)
                    # prefix_embds = embd(prefix)

                    # Mask the tokens before the first token of the target   # MASK  
                    target_mask = torch.zeros_like(prompts_targets_ids).bool() 
                    target_mask[..., :label_mask.size(1)] = torch.repeat_interleave(label_mask, hparams.num_return_sequences, dim=0)


                    # Mask the tokens after eos token for completions
                    loss_mask = (target_mask)

                    # Compute the log probabilities of the prompts_targets_ids, make sure to align the prompts_targets_ids with the logits:
                    # 1. Shift the prompts_targets_ids by dropping the first token.
                    '''
                    sequences这里是否有抛弃第一个token的必要?感觉好像不需要抛弃？
                    '''
                    shift_sequences = prompts_targets_ids[..., 1:].contiguous()   
                    shift_loss_mask = loss_mask[..., 1:].contiguous()
                    # shift_target_mask = target_mask[..., 1:].contiguous()
                    # shift_completion_mask = completion_mask[..., 1:].contiguous()

                    
                    # 2. Shift the probabilites by dropping the last token.
                    from ...prefix_editor.utils  import register_hook,register_attn_hook,register_token_hook,register_midattn_hook,register_subject_hook
                    
                    query_last_token = []
                    # query_last_ffn_token = []
                    # query_last_attn_token = []
                    # query_mid_attn_token = []
                    subject_attn_token = []
                    # handle= register_hook(model,prompts_ids,hparams,query_last_ffn_token)
                    # handle2= register_attn_hook(model,prompts_ids,hparams,query_last_attn_token)
                    # handle3=register_midattn_hook(model,prompts_ids,prompts_targets_ids,hparams,query_mid_attn_token)
                    
                    # print(f"start_end: {start_end}")
                    handle4=register_subject_hook(model,prompts_ids,hparams,subject_attn_token,start_end)
                    # out = model(input_ids=prompts['input_ids'],past_key_values=PAD_past_kv,output_attentions=True)
                    out = model(input_ids=prompts_ids,output_attentions=True)
                    # handle.remove()
                    # handle2.remove()
                    # handle3.remove()
                    handle4.remove()
                    # query_last_token.append(torch.cat((query_last_ffn_token[0]*0,query_last_attn_token[0]),dim=-1))
                    # query_last_token.append(torch.cat((query_last_attn_token[0]*0,query_mid_attn_token[0]*1),dim=-1))
                    query_last_token.append(torch.cat((subject_attn_token[0]*0,subject_attn_token[0]*1),dim=-1))
                    if len(last_token_hd)==0:  last_token_hd.append(query_last_token[0])
                    
                    
                    tok_handle = register_token_hook(model,prefix_embds,hparams,device)
                    shift_logits = model(input_ids=prompts_targets_ids).logits[:, :-1, :]
                    # shift_logits = model(input_ids=prompts_targets_ids,past_key_values=PAD_past_kv).logits[:, :-1, :]
                    tok_handle.remove()
                    
                    # shift_logits = model(input_ids=prompts_targets_ids,past_key_values=new_past_kv).logits[:, :-1, :]
                    shift_q = torch.softmax(shift_logits, dim=-1).contiguous()
                    # shift_q = torch.gather(shift_q, -1, shift_sequences.unsqueeze(-1)).squeeze(-1)
                    shift_log_q = torch.log(shift_q)    # cross_entropy实现原理 me
                    
                    if hparams.print_kl:pass  # True  me
                    if hparams.print_kl==False:
                        mle_loss = -torch.gather(shift_log_q, -1, shift_sequences.unsqueeze(-1)).squeeze(-1)  
                        loss = (shift_loss_mask * mle_loss).sum(-1) / shift_loss_mask.sum(-1)
                        # answers = torch.argmax(shift_logits, dim=-1)
                        # print(f"labels: {shift_sequences}")
                        # print(f"labels: {shift_sequences.shape}")
                        # print(f"out_answers: {answers}")
                        # print(f"shift_q: {shift_q.shape}")
                        # ans_indice = torch.where()
                        # print(f"seeds: {seeds}")
                        
                        
                    loss = loss.mean()
                else:
                    raise NotImplementedError
            # print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=batchsize)

            # if loss.item() >= 1e-2 or hparams.print_kl:
            # loss.requires_grad_(True)
            loss.backward()
            opt.step()

            # print(f"LR: {scheduler.get_last_lr()[0]}")
            # scheduler.step()


        # print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2 and not hparams.print_kl:
            break

    deltas = None
    print(f"Deltas successfully computed for {list(weights.keys())}")
    return last_token_hd,subject_attn_token,embd_list,deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
