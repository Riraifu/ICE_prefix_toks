import json
# path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_8000_10000.json'
# path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_10000.json'
path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_10000_origin.json'

# out = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_2000_4000.json'
with open(path,'r') as file:
    datas = json.load(file)

from easyeditor.prefix_editor.utils  import get_start_end,find_subject
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
model_name = "/home/zhuliyi/LLM_bases/gpt2"
# model_name = "/home/zhuliyi/LLM_bases/Llama-2-7b-hf"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token_id = tok.eos_token_id
trained_fusion_ids = {}
device = "cuda:5"
for idx,data in enumerate(datas):
    _subject = ' '+data['subject']
    _subject_ids = tok(_subject,padding=True,return_tensors='pt')['input_ids'].to(device)
    _subject_ids_list = [str(id) for id in _subject_ids[0].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
    _subject_ids_str = '_'.join(_subject_ids_list)
    trained_fusion_ids[_subject_ids_str]=_subject
    
    subject = data['subject']
    subject_ids = tok(subject,padding=True,return_tensors='pt')['input_ids'].to(device)
    # subject_ids_list = [str(id) for id in subject_ids[0].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
    subject_ids_list = [str(id) for id in subject_ids[0][1:].cpu().tolist() if id!=tok.pad_token_id]  # 仅支持batchsize=1
    subject_ids_str = '_'.join(subject_ids_list)
    trained_fusion_ids[subject_ids_str]=subject
    
    print(f"idx: {idx}")
    
    origin_prompt = data['prompt']
    origin_prompt_ids = tok(origin_prompt,padding=True,return_tensors='pt')['input_ids'].to("cuda:3")
    print(f"origin_prompt: {origin_prompt}")
    origin_subject_ids,start_end = find_subject(origin_prompt_ids, trained_fusion_ids)
    origin_subject_ids = origin_subject_ids.to("cuda:3")
    
    prompt = data['portability']['Generality'][0]['prompt']
    prompt_ids = tok(prompt,padding=True,return_tensors='pt')['input_ids'].to(device)
    print(f"prompt: {prompt}")
    print(f"prompt_ids: {prompt_ids}")
    subject_ids,start_end = find_subject(prompt_ids, trained_fusion_ids)
    subject_ids = subject_ids.to(device)