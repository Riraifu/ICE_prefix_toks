import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer #, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
# from accelerate import Accelerator
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import pre_compute_edit_quality,compute_edit_quality, compute_edit_quality2, compute_icl_edit_quality, compute_sent_metric,get_gen_sent,get_gen_sent2
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *


# from transformers import GPTJForCausalLM

from ..prefix_editor.modeling_gpt2 import GPT2LMHeadModel
from ..prefix_editor.modeling_llama import LlamaForCausalLM
from ..prefix_editor.modeling_gptj import GPTJForCausalLM

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")
        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            # elif 'gpt-3.5' in self.model_name.lower():
            #     self.model, self.tok = None, None
            elif 'gpt-j' in self.model_name.lower():
                self.model = GPTJForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id   # GPTJ的eos_token_id和unk_token_id 一样 都是50256

            elif 'gpt2' in self.model_name.lower():
                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                # self.model = ModifiedGPT2.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                # self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                # self.tok = GPT2TokenizerFast.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id   # GPT2的eos_token_id和unk_token_id 一样 都是50256

            elif 'llama' in self.model_name.lower():
                # self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id              # Llama2的eos_token_id=2     unk_token_id=1
                print(f"self.tok.unk_token_id: {self.tok.unk_token_id}")   
                print(f"self.tok.pad_token_id: {self.tok.pad_token_id}")
                # self.tok.padding_side = 'left'
            # elif 'baichuan' in self.model_name.lower():
            #   ``  self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'chatglm' in self.model_name.lower():
            #     self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.unk_token_id = 64787
            #     # self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'internlm' in self.model_name.lower():
            #     self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'qwen' in self.model_name.lower():
            #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            # elif 'mistral' in self.model_name.lower():
                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                # self.tok = AutoTokenizer.from_pretrained(self.model_name)
                # self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            

            
            if 'gpt2' in self.model_name.lower() or 'gpt-j' in self.model_name.lower():
                self.tok.padding_side = 'left'
            elif 'llama' in self.model_name.lower():
                self.tok.padding_side = 'right'
            # if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'ICE']):
            #     LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            #     self.tok.padding_side = 'left'
            # if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'ICE']):
            #     LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            #     self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics = False, 
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False           # True  me
        save_gen_sentence = kwargs['save_gen_sentence'] if 'save_gen_sentence' in kwargs.keys() else False     # True  me
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        # if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
        #     self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]     # 默认为<|endoftext|>  

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:  # √
            requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                            locality_inputs, portability_inputs, **kwargs)
        # if hasattr(self.hparams, 'batch_size') :
        #        assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")


        all_metrics = []
        gen_sentence = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:  # pre_edit文件已存在
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:   # pre_edit文件不存在，则计算metric并导入新pre_edit文件
            # for i, request in enumerate(tqdm(self._chunks(requests,self.hparams.batch_size))):
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    pass
                    # assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    # metrics = {
                    #     "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                    #                                     request, self.hparams.device, pre_edit=True)
                    # }
                else:
                    metrics = {
                        "pre": pre_compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                self.hparams.device, test_generation=test_generation)
                    }  # 计算ppl、acc、locality、portability、fluency
                all_metrics.append(metrics)
                    # print(f"666request: {len(request)}")
                    # ret = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                    #                             self.hparams.device, test_generation=test_generation)
                    # for k in range(len(request)):
                    #     metrics = {"pre": ret[k]}  # 计算ppl、acc、locality、portability、fluency
                    #     all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)
        
        
        import faiss
        prefix_len = 1
        self.index  = faiss.IndexFlatIP(self.hparams.n_embd*2) # IndexFlatIP内积距离、IndexFlatL2欧氏距离
        self.index2 = faiss.IndexFlatIP(self.hparams.n_embd) # IndexFlatIP内积距离、IndexFlatL2欧氏距离
        # self.index = faiss.IndexFlatL2(self.hparams.n_embd*prefix_len) # IndexFlatIP内积距离、IndexFlatL2欧氏距离
        vector_base = []
        # self.model.transformer.h[self.hparams.n_layer-2].attn = GPT2Attn  # 修改_attn

        trained_fusion_ids = {'match':set(),
                            'x_match':set(),'x_x_match':set(),
                            'match_x':set(),'match_x_x':set()}
        for i, request in enumerate(tqdm(self._chunks(requests,self.hparams.batch_size))):
            start = time()
            if self.alg_name == 'FT':
                # edited_model, weights_copy = self.apply_algo(
                
                last_token_hd,subject_attn_token,embd_list,edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    # [request],
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    trained_fusion_ids=trained_fusion_ids,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                # continue
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                prefix = torch.arange(start=0,end=prefix_len).to(self.hparams.device) #
                prefix_list = []
                batchsize = len(request)
                for k in range(batchsize):
                    prefix_list.append(embd_list[k](prefix))
                prefix_embds = torch.cat(prefix_list,dim=0)

                # prefix_embds = prefix_embds.clone().detach().view(1,-1)
                # norm = torch.norm(prefix_embds)
                # prefix_embds_norm = (prefix_embds/norm).detach().cpu().numpy()
                
                for k in range(batchsize):
                    cur_last_token_hd =  last_token_hd[0][k]
                    cur_last_token_hd_norm = (cur_last_token_hd/torch.norm(cur_last_token_hd)).unsqueeze(dim=0).detach().cpu().numpy()
                    # print(f"cur_last_token_hd_norm[{k}]:{cur_last_token_hd_norm.shape}")
                    self.index.add(cur_last_token_hd_norm) 
                    # self.index.add(cur_last_token_hd.unsqueeze(dim=0).detach().cpu().numpy()) 
                    vector_base.append(prefix_embds[k].unsqueeze(dim=0).clone().detach().cpu().numpy())
                    # print(f"prefix_embds[k]: {prefix_embds[k].unsqueeze(dim=0).shape}")
                # print(f"last_token_hd: {torch.mean(last_token_hd[0], dim=0).unsqueeze(dim=0).shape}")
                # last_token_hd = last_token_hd[0][0,...].detach().unsqueeze(dim=0)
                # last_token_hd = torch.mean(last_token_hd[0].clone().detach(), dim=0).unsqueeze(dim=0).detach()
                # last_token_hd_norm = (last_token_hd/torch.norm(last_token_hd)).detach().cpu().numpy()
                # print(f"query_mid_attn_token: {len(query_mid_attn_token)}")
                # for k in range(batchsize):
                #     cur_subject = subject_attn_token[0][k]
                #     cur_subject_norm = (cur_subject/torch.norm(cur_subject)).unsqueeze(dim=0).detach().cpu().numpy()
                #     self.index2.add(cur_subject_norm) 
                    
                    
                    # cur_midattn =  query_mid_attn_token[0][k]
                    # cur_midattn_norm = (cur_midattn/torch.norm(cur_midattn)).unsqueeze(dim=0).detach().cpu().numpy()
                    # self.index2.add(cur_midattn_norm) 
                    # print(f"cur_midattn_norm[{k}]:{cur_midattn_norm.shape}")
                    

                # self.index.add(prefix_embds_norm) 
                # self.index.add(prefix_embds_norm+last_token_hd_norm) 
                # vector_base.append(prefix_embds.clone().detach().cpu().numpy())
                # query_vector = None
                # distances, indices = self.index.search(query_vector, k)
                print(f"=================   self.index.ntotal: {self.index.ntotal}  =================")

                start = time()
                ret = compute_edit_quality2(embd_list,edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation,
                                                  index=self.index, index2=None,vector_base=vector_base,trained_fusion_ids=trained_fusion_ids) # 
                for k in range(batchsize):
                    data_idx = i*self.hparams.batch_size+k
                    all_metrics[data_idx].update({
                        'case_id': data_idx,
                        "requested_rewrite": request[k],
                        "time": exec_time,
                        # "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                        "post": ret[k]
                    }) # 计算edit后的指标
                
                # print(f"epoch {i}: {ret}")
                # print(f"epoch {i}: "ret[j])
                # continue
                
                # sent = get_gen_sent(edited_model,self.tok,request,test_generation = test_generation,save_gen_sentence = save_gen_sentence)
                # for k in range(batchsize):
                #     sent = get_gen_sent2(embd_list[k],self.hparams,edited_model,self.tok,request[k],test_generation = test_generation,save_gen_sentence = save_gen_sentence,device=self.hparams.device,index=self.index,vector_base=vector_base,metric_type="generate_fast")
                #     gen_sentence.append(sent)

                # continue
                # if "metric_kwargs" in kwargs:
                #     all_metrics[i].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok, metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
                # continue
                # if self.alg_name == 'KN' or (self.alg_name == 'GRACE' and keep_original_weight):
                #     with torch.no_grad():
                #         weights_copy() # unpatch_fn
                # elif self.alg_name == 'LoRA' and  keep_original_weight:
                #     edited_model.unload()
                #     del self.model.peft_config
                # elif self.alg_name == 'MELO':
                #     self.model = edited_model
                # elif self.alg_name == 'LoRA' and not keep_original_weight:
                #     self.model = edited_model
                # else:
                #     with torch.no_grad():
                #         for k, v in weights_copy.items():
                #             nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                
                # for j in range(batchsize):  
                #     data_idx = i*self.hparams.batch_size+j
                #     if 'locality' in all_metrics[data_idx]['post'].keys():
                #         for locality_key in request[j]['locality'].keys():
                #             assert len(all_metrics[data_idx]['post']['locality'][f'{locality_key}_output']) == \
                #                 len(all_metrics[data_idx]['pre']['locality'][f'{locality_key}_output'])
                #             locality_result = []
                #             for ans,label in zip(all_metrics[data_idx]['post']['locality'][f'{locality_key}_output'],all_metrics[data_idx]['pre']['locality'][f'{locality_key}_output']):
                #                 locality_result.append(np.mean(np.equal(ans, label)))
                #             all_metrics[data_idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                #             all_metrics[data_idx]['post']['locality'].pop(f'{locality_key}_output')
                #         all_metrics[data_idx]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                from ..prefix_editor.utils import cal_acc
                cal_acc(all_metrics,i*self.hparams.batch_size+batchsize)

                # if verbose:
                #     for j in range(batchsize): 
                #         data_idx = i*self.hparams.batch_size+j 
                #         LOG.info(f"{data_idx} editing: {request[j]['prompt']} -> {request[j]['target_new']}  \n {all_metrics[data_idx]}")
                #         print('*****************************')
            


            editing_method = kwargs['editing_method']
            datatype = kwargs['datatype']
            if self.hparams.alg_name in ['MEMIT','ROME','ICE','FT']:
                if not os.path.exists(f'./outputs/{self.hparams.model_name.split("/")[-1]}/{editing_method}'):
                    os.makedirs(f'./outputs/{self.hparams.model_name.split("/")[-1]}/{editing_method}')
                with open(f'./outputs/{self.hparams.model_name.split("/")[-1]}/{editing_method}/GPT2_attn_token_{editing_method}_{datatype}_{self.hparams.model_name.split("/")[-1]}_gen_sentence.json', 'w') as wf:
                    json.dump(gen_sentence, wf,indent=4)
            
            start = i*self.hparams.batch_size
            end = start+batchsize
            # for kk in range(start,end):
            #     # print(f"kk: {kk}")
            #     json.dump(metrics[kk], open(os.path.join(f"./results/{self.hparams.model_name.split('/')[-1]}/ICE", f'llama2_p01{editing_method}_{datatype}_{self.hparams.model_name.split("/")[-1]}_results.json'), 'a'), indent=4)
            json.dump(all_metrics, open(os.path.join(f"./results/{self.hparams.model_name.split('/')[-1]}/ICE", f'GPT2_attn_token_{editing_method}_{datatype}_{self.hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)


            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        # if isinstance(edited_model, LORA):
        #     edited_model=edited_model.model
        #for melo
        
        if summary_metrics and len(all_metrics)!=0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            logs_dir = './logs'  
            if not os.path.exists(logs_dir):  
                os.makedirs(logs_dir)  
            output_file = os.path.join(logs_dir, 'results.json')
            with open(output_file, 'w') as f:  
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)
            
            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean([metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
            
            print("Metrics Summary: ", mean_metrics)

        return all_metrics, edited_model, weights_copy




    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    # me   request为一个样本  <prompt,target_new,ground_truth,context,subject,portability,locality>
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'context' in kwargs:
            for i, request in enumerate(requests):
                request.update({'context': kwargs['context'][i]}                )

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        # print(f"locality_inputs:::: {locality_inputs}")

        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                print(f"locality_key:::::{locality_key}")
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                print(f"len(locality_inputs[locality_key]['prompt']): {len(locality_inputs[locality_key]['prompt'])}")
                print(f"len(locality_inputs[locality_key]['ground_truth']: {len(locality_inputs[locality_key]['ground_truth'])}")
                print(f"len(requests): {len(requests)}")
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    if portability_inputs[portability_key]['prompt'][i] is not None:
                        request['portability'].update(
                            {
                                portability_key: {
                                    'prompt': portability_inputs[portability_key]['prompt'][i],
                                    'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                                }
                            }
                        )
        return requests

    def edit_requests(self,
             requests,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        for i, request in enumerate(tqdm(requests)):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                }
            all_metrics.append(metrics)

        for i, request in enumerate(tqdm(requests)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
                all_metrics[i]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )

            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if self.alg_name == 'KN' or self.alg_name == 'GRACE':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                elif self.alg_name == 'LoRA' and keep_original_weight:
                    edited_model.unload()
                    del self.model.peft_config
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                if 'locality' in all_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans,label in zip(all_metrics[i]['post']['locality'][f'{locality_key}_output'],all_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[i]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        return all_metrics, edited_model, weights_copy
