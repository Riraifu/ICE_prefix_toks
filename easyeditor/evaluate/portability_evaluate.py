from transformers import AutoTokenizer
from ..util import HyperParams
from typing import List
import typing
import torch
import numpy as np
from .evaluate_utils import  test_batch_prediction_acc, test_seq2seq_batch_prediction_acc, test_prediction_acc,test_prediction_acc2


def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret

def compute_portability_quality2(
    embd_list,
    idx,
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
    index = None,
    index2= None,
    vector_base=[],
    requests = None,
    metric_type = None,
    trained_fusion_ids={}
) -> typing.Dict:


    from ..prefix_editor.utils  import get_start_end,find_subject
    prefix_len=1

    start_end = []
    print(f"prompt: {prompt}  {type(prompt)}")
    prompt_ids = tok(prompt, return_tensors="pt", padding=True)['input_ids'].to(device)
    batchsize = prompt_ids.shape[0]
    PAD = torch.tensor(tok.pad_token_id).to(device).repeat(batchsize,prefix_len)
    prompt_ids = torch.cat((PAD,prompt_ids),dim=1)

    subject_ids,start_end = find_subject(prompt_ids, trained_fusion_ids)
    if subject_ids is not None:subject_ids = subject_ids.to(device)
    
    
    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        # portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')
        portability_correct,dist,indices,threshold = test_prediction_acc2([embd_list[idx]],model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE',index=index,index2=index2,vector_base=vector_base,metric_type=metric_type,start_end=start_end)

    ret = {
        f"{portability_key}_acc": portability_correct,
        f"{portability_key}_dist": str(dist),
        f"{portability_key}_indices": str(indices),
        f"{portability_key}_threshold": threshold
    }
    return ret