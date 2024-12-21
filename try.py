

DDict = {"____nihao_11": "666",
         "333_nihao":"777",
         "444_nihao_222":"888",
         "nihao_222":"888",
         "nihao_222":"888",
         "nihao_222_3333":"888",
         "nihao":'8888999'}
# print("nihao_ya" in DDict)
# print("nihao" in DDict)
import re
key = 'nihao'
pattern = rf'^[\d_|\d_\d_]*{key}[_\d|_\d_\d]*$'  # 


# print(f"DDict[-1]:{list(DDict.items())[-1]}")
for k,v in DDict.items():
    print(re.match(pattern, k))
    start = re.match(pattern, k).span()[0]
    end = re.match(pattern, k).span()[1]-1
    print(start,end)
    


exit()

# import json
# path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_10000_copy.json'
# out = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_2000_4000.json'
# with open(path,'r') as file:
#     datas = json.load(file)

# with open(out,'w') as f:
#     json.dump(datas[2000:4001], f, indent=4)


from transformers import GPT2TokenizerFast, AutoTokenizer,LlamaForCausalLM
# tokenizer = GPT2TokenizerFast.from_pretrained('/home/zhuliyi/LLM_bases/gpt2')
tokenizer = AutoTokenizer.from_pretrained('/home/zhuliyi/LLM_bases/Llama-2-7b-hf')
# model = LlamaForCausalLM.from_pretrained('/home/zhuliyi/LLM_bases/Llama-2-7b-hf').to('cuda:4')
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eos_token_id


generation_prompts = ["What type of submarine was SM UB-128 classified as?",
                      "What kind of water unit is SM UB-128?",
                      "Which year did Da Chu end?",
                      "What year has Da Chu been dissolved?"]


generation_prompts = ["What does Pont du Garigliano cross over?",
                      "Which year did Da Chu end?",
                      "At what location did Enno Littmann die?",
                      "What country released Daft Planet?",
                      "What year did Da Chu end?",
                      "What city is KPFT located?",
                      "What city or city serves KPFT?"
                      ]


generation_prompts = ["What town or city does KBOV serve?",
                      "Which ailment caused the death of J\u00falio Dinis?",
                      "What is the constellation where NGC 6188 is located?",
                      "What country released Daft Planet?",
                      "Of which constellation is Messier 10 a part?",
                      "What is the constellation that is made with HD 150706?",
                      "What city or city serves KPFT?"
                      ]
generation_prompts = [
    ["Michael Stroka"],
    [" Michael Stroka"],
    ["What kind of medical issue did Michael Stroka have?"],
    ["What disease did Michael Stroka kill?"],
    ["What was Michael Holm's career?"],
    ["What was Michael Holmes' career?"]
]
batch1 = tokenizer(generation_prompts[0], return_tensors='pt', padding=True).to('cuda:4')
batch2 = tokenizer(generation_prompts[1], return_tensors='pt', padding=True).to('cuda:4')
batch3 = tokenizer(generation_prompts[2], return_tensors='pt', padding=True).to('cuda:4')
batch4 = tokenizer(generation_prompts[3], return_tensors='pt', padding=True).to('cuda:4')
batch5 = tokenizer(generation_prompts[4], return_tensors='pt', padding=True).to('cuda:4')
batch6 = tokenizer(generation_prompts[5], return_tensors='pt', padding=True).to('cuda:4')

# # # generated_ids = model.generate(batch['input_ids'], max_length=22,top_k = 1) # , length_penalty=2.0  
print(f"batch1: {batch1['input_ids']}")
print(f"batch2: {batch2['input_ids']}")
print(f"batch3: {batch3['input_ids']}")
print(f"batch4: {batch4['input_ids']}")
# print(f"batch5: {batch5['input_ids']}")
# print(f"batch6: {batch6['input_ids']}")
# print(f"generated_ids: {generated_ids}")
# print(f"generated_ids: {generated_ids.shape}")
# for ids in batch:
#     print(tokenizer.decode(ids, skip_special_tokens=False))
#     print('-===========')



from transformers import GPT2TokenizerFast, LlamaForCausalLM
tokenizer = GPT2TokenizerFast.from_pretrained('/home/zhuliyi/LLM_bases/gpt2')
# model = LlamaForCausalLM.from_pretrained('/home/zhuliyi/LLM_bases/Llama-2-7b-hf').to('cuda:4')
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eos_token_id



# from collections import defaultdict
# fusion_ids_trained = {}
# generation_prompts = [
#     "Michael Holm",
#     " Michael Holm",
#     "Michael Holm's",
#     " Michael Holm's",
#     "What was Michael Holm's career?",
#     "What was Michael Holmes' career?"
# ]
# # generation_prompts="Michael Holm"
# subject_ids = tokenizer(generation_prompts, return_tensors='pt', padding=True).to('cuda:4')
# batch2 = tokenizer(generation_prompts[1], return_tensors='pt', padding=True).to('cuda:4')
# batch3 = tokenizer(generation_prompts[2], return_tensors='pt', padding=True).to('cuda:4')
# batch4 = tokenizer(generation_prompts[3], return_tensors='pt', padding=True).to('cuda:4')
# batch5 = tokenizer(generation_prompts[4], return_tensors='pt', padding=True).to('cuda:4')
# batch6 = tokenizer(generation_prompts[5], return_tensors='pt', padding=True).to('cuda:4')

# print(f"subject_ids: {subject_ids['input_ids']}")
# print(f"batch2: {batch2['input_ids']}")
# print(f"batch3: {batch3['input_ids']}")
# print(f"batch4: {batch4['input_ids']}")
# print(f"batch5: {batch5['input_ids']}")
# print(f"batch6: {batch6['input_ids']}")
# print('##################')
# print(f"subject_ids['input_ids']: {subject_ids['input_ids']}")

# for input_ids in subject_ids['input_ids']:
#     print(f"input_ids: {input_ids}")
# subject_ids = subject_ids['input_ids'][0]
# subject_ids_list = subject_ids.tolist()

# print(f"subject_ids_list: {subject_ids_list}")
# subject_ids_list = [str(i) for i in subject_ids_list if i!=tokenizer.pad_token_id]
# fusion_id = '_'.join(subject_ids_list)
# print(f"fusion_idAA: {fusion_id}")
# print(f"subject_ids: {subject_ids['input_ids'][0].shape}")
# fusion_ids_trained[fusion_id]=subject_ids['input_ids'][0].cpu().numpy().tolist()
# print(f"fusion_ids_trained: {fusion_ids_trained}")

# if '13256_6479_76'  in fusion_ids_trained:
# #     print(True)

# pattern = r'\d+'  # 匹配一个或多个数字
# text = '123abc'

# match = re.match(pattern, text)
# if match:
#     print("匹配成功:", match.group())  # 输出匹配到的内容
# else:
#     print("匹配失败")


exit()













# import torch
# x=1
# y=768
# shape = (x,y)
# p=0.9
# probs = torch.full(shape, p)
# mask = torch.bernoulli(probs)
# # mask = torch.rand(shape) < p
# print(mask==1)
# print(sum(sum(mask))/(x*y))

# exit()


# import nlpaug.augmenter.word as naw
# from nlpaug.flow import Sometimes

# # 增强时，会保持下面列表中的内容不变。
# stopwords = ["love", "i"]
# synonym_aug = naw.SynonymAug(stopwords=stopwords)
# spelling_aug = naw.SpellingAug(stopwords=stopwords, aug_p=0.1)
# # 将多种数据增强方式融合
# aug = Sometimes([synonym_aug, spelling_aug])
# text = "i love apple. i was born in 2000. how are you?"
# r = aug.augment(text, 2)
# import torch
# import torch.nn.functional as F
# z = torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]])
# probabilities = F.softmax(z, dim = 1)
# print(probabilities)
# print(torch.argmax(probabilities,dim=1))

# lists =  [[] for _ in range(50257)]
# # 不同list 用字典
# print(len(lists))
# print(lists)

# exit()
# from pygtrans import Translate
#经过测试，这个翻译的包翻译的时间是最短的

# client = Translate(proxies={'https': 'http://localhost:10809'})

# # 检测语言
# text = client.detect('Answer the question.')
# assert text.language == 'en'

# # 翻译句子
# text = client.translate('Look at these pictures and answer the questions.')
# assert text.translatedText == '看这些图片，回答问题。'

# # 批量翻译
# texts = client.translate([
#     'Good morning. What can I do for you?',
#     'Read aloud and underline the sentences about booking a flight.',
#     'May I have your name and telephone number?'
# ])
# assert [text.translatedText for text in texts] == [
#     '早上好。我能为你做什么？',
#     '大声朗读并在有关预订航班的句子下划线。',
#     '可以给我你的名字和电话号码吗？'
# ]

# # 翻译到日语
# text = client.translate('请多多指教', target='ja')
# assert text.translatedText == 'お知らせ下さい'

# # 翻译到韩语
# text = client.translate('请多多指教', target='ko')
# assert text.translatedText == '조언 부탁드립니다'

# # 文本到语音
# tts = client.tts('やめて', target='ja')
# open('やめて.mp3', 'wb').write(tts)
# exit()
# from transformers import trainer

# import torch.nn.functional as F
# F.cross_entropy()
# exit()

import json



# 读取JSON文件
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/GS_LS_eval_10_ICE_zsre_Llama-2-7b-hf_results.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/gpt2/ICE/GS_LS_eval_10_ICE_zsre_gpt2_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama_avg_ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama_avg_2ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama_avg_-3ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/result`s/Llama-2-7b-hf/ICE/llama_avg_-8ICE_zsre_Llama-2-7b-hf_results.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_tryICE_zsre_Llama-2-7b-hf_results.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p90ICE_zsre_Llama-2-7b-hf_results.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p70ICE_zsre_Llama-2-7b-hf_results.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p50ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p30ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p10ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/Llama-2-7b-hf/ICE/llama2_p05ICE_zsre_Llama-2-7b-hf_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/gpt2/ICE/llama2_p01ICE_zsre_gpt2_results.json"
path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/gpt2/ICE/GPT2_attn_token_ICE_zsre_gpt2_results.json"

with open(path, 'r') as file:
    datas = json.load(file)

# print(len(datas))
# print(datas[0])
num = len(datas)


# ES_index2 = [0]*11
# GS_index2 = [0]*11
# LS_index2 = [0]*11
# kedu2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,100]


ES_index2 = [0]*21
GS_index2 = [0]*21
LS_index2 = [0]*21
kedu2 = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,100]

import re
# 0.0<=x<0.1  0.1<=0.2    0.9<=x<=1.0

for data in datas:
    A = data['post']
    # ES_dist_max = re.sub(r"\s+", ",", data['post']['rewrite_dist'].replace('  ',','))
    ES_dist_max = float(data['post']['rewrite_dist'].split(' ')[0][2:].replace(']',''))
    # if 0.0<=ES_dist_max<0.1: ES_index[0]+=1
    # elif 0.1<=ES_dist_max<0.2: ES_index[1]+=1
    # elif 0.2<=ES_dist_max<0.3: ES_index[2]+=1
    # elif 0.3<=ES_dist_max<0.4: ES_index[3]+=1
    # elif 0.4<=ES_dist_max<0.5: ES_index[4]+=1
    # elif 0.5<=ES_dist_max<0.6: ES_index[5]+=1
    # elif 0.6<=ES_dist_max<0.7: ES_index[6]+=1
    # elif 0.7<=ES_dist_max<0.8: ES_index[7]+=1
    # elif 0.8<=ES_dist_max<0.9: ES_index[8]+=1
    # elif 0.9<=ES_dist_max<=1.0: ES_index[9]+=1
    # elif 1.0<ES_dist_max: ES_index[10]+=1


    for i in range(len(kedu2)-1):
        if kedu2[i]<=ES_dist_max<kedu2[i+1]:
            ES_index2[i]+=1
            # print(f"{kedu2[i]}<=ES_dist_max<{kedu2[i+1]}")
            break
    
    # # print(A)
    GS_dist_max = float(data['post']['portability']['Generality_dist'].split(' ')[0][2:].replace(']',''))
    for i in range(len(kedu2)-1):
        if kedu2[i]<=GS_dist_max<kedu2[i+1]:
            GS_index2[i]+=1
            # print(f"{kedu2[i]}<=ES_dist_max<{kedu2[i+1]}")
            break
    # if 0.0<=GS_dist_max<0.1: GS_index[0]+=1
    # elif 0.1<=GS_dist_max<0.2: GS_index[1]+=1
    # elif 0.2<=GS_dist_max<0.3: GS_index[2]+=1
    # elif 0.3<=GS_dist_max<0.4: GS_index[3]+=1
    # elif 0.4<=GS_dist_max<0.5: GS_index[4]+=1
    # elif 0.5<=GS_dist_max<0.6: GS_index[5]+=1
    # elif 0.6<=GS_dist_max<0.7: GS_index[6]+=1
    # elif 0.7<=GS_dist_max<0.8: GS_index[7]+=1
    # elif 0.8<=GS_dist_max<0.9: GS_index[8]+=1
    # elif 0.9<=GS_dist_max<=1.0: GS_index[9]+=1
    # elif 1.0<GS_dist_max: GS_index[10]+=1

    print(data['post']['locality']['Locality_dist'].split(' ')[0][2:].replace(']',''))
    print('-----')
    LS_dist_max = float(data['post']['locality']['Locality_dist'].split(' ')[0][2:].replace(']',''))
    for i in range(len(kedu2)-1):
        if kedu2[i]<=LS_dist_max<kedu2[i+1]:
            LS_index2[i]+=1
            # print(f"{kedu2[i]}<=ES_dist_max<{kedu2[i+1]}")
            break
    # if 0.0<=LS_dist_max<0.1: LS_index[0]+=1
    # elif 0.1<=LS_dist_max<0.2: LS_index[1]+=1
    # elif 0.2<=LS_dist_max<0.3: LS_index[2]+=1
    # elif 0.3<=LS_dist_max<0.4: LS_index[3]+=1
    # elif 0.4<=LS_dist_max<0.5: LS_index[4]+=1
    # elif 0.5<=LS_dist_max<0.6: LS_index[5]+=1
    # elif 0.6<=LS_dist_max<0.7: LS_index[6]+=1
    # elif 0.7<=LS_dist_max<0.8: LS_index[7]+=1
    # elif 0.8<=LS_dist_max<0.9: LS_index[8]+=1
    # elif 0.9<=LS_dist_max<=1.0: LS_index[9]+=1
    # elif 1.0<LS_dist_max: LS_index[10]+=1
print(f"ES:{ES_index2}")
print(f"GS:{GS_index2}")
print(f"LS:{LS_index2}")
# print(f"ES:{ES_index}")
# print(f"GS:{GS_index}")
# print(f"LS:{LS_index}")
# assert 1==1
# prompts = 11
# target_new=111
# prompts, target_new = [prompts,], [target_new,]
# print(prompts)
# print(target_new)

# for i in range(self.hparams.batch_size):
#     cur_last_token_hd =  last_token_hd[0][i]
#     cur_last_token_hd_norm = (cur_last_token_hd/torch.norm(cur_last_token_hd)).detach().cpu().numpy()
#     print(f"cur_last_token_hd_norm[{i}]:{cur_last_token_hd_norm.shape}")
#     self.index.add(cur_last_token_hd_norm) 
#     vector_base.append(prefix_embds[i])
#     print(f"prefix_embds: {prefix_embds.shape}")
#     print(f"prefix_embds[i]: {prefix_embds[i].shape}")
# import json
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/results/gpt2/ICE/GS_LS_eval_10_ICE_zsre_gpt2_results.json"
# with open(path, 'r') as f:
#     result = json.load(f)
# print(len(result))







exit()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
model_name = "/home/zhuliyi/LLM_bases/gpt2"
prompts_targets = "What university did Watts Humphrey attend?"

tok = AutoTokenizer.from_pretrained(model_name)
tok.padding_side = 'left'
tok.pad_token_id = tok.eos_token_id
prompts_targets = tok(prompts_targets, return_tensors="pt", padding='max_length', max_length=32).to("cuda:0")
print(prompts_targets)


exit()


# from transformers import trainer

import torch
from time import time

def find_subarray_index(input_ids, subject_ids):
    """
    函数用于查找子数组subject_ids在input_ids中的起始索引位置，这里假设input_ids和subject_ids是torch.Tensor类型
    """
    input_len = input_ids.size(0)
    sub_len = subject_ids.size(0)
    for i in range(input_len - sub_len + 1):
        if torch.equal(input_ids[i:i + sub_len], subject_ids):
            return i
    return -1  # 如果没找到，返回-1表示不存在


# 示例用法
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
subject_ids = torch.tensor([[3, 4]])
start_time = time()
index = find_subarray_index(input_ids[0], subject_ids[0])
end_time = time()
print(index,end_time-start_time)
print(input_ids.shape)
print(subject_ids.shape)

def get_start_end(M_name,input_ids,subject_ids):
    if 'gpt2' in M_name or 'gpt2-xl' in M_name or "gpt-j-6b" in M_name:
        subject_ids = subject_ids
    elif "llama-2-7b" in M_name:
        subject_ids = subject_ids[:,1:]

    start_end = []
    _subject_len = subject_ids.shape[1]
    for i in range(len(input_ids)):  # 遍历A中每个ids串
        id_seq = input_ids[i]
        target_seq = subject_ids[i]
        target_len = len(target_seq)
        for j in range(len(id_seq) - target_len + 1):  # 遍历可能的起始位置
            if torch.equal(id_seq[j:j + target_len], target_seq):  # 比较切片是否和目标字串相等
                start_end.append([j,j+_subject_len-1])
                # start_end.append([j+prefix_len,j+_subject_len-1+prefix_len])
                break  # 找到就跳出内层循环，进入下一个ids串与对应字串的匹配
    # print(f"input_ids: {input_ids.shape}")
    # print(f"subject_ids: {subject_ids.shape}")
    # print(f"input_ids: {input_ids}")
    # print(f"subject_ids: {subject_ids}")
    return start_end

start_time = time()
start_end = get_start_end("gpt2",input_ids,subject_ids)
end_time = time()
print(start_end,end_time-start_time)
exit()



# import torch
# import torch.nn as nn
# embd = nn.Embedding(1,768)

# weights = {
#     n: p
#     for n, p in embd.named_parameters()
# }
# # print(weights)
# # for n,w in embd.named_parameters():
# #     w.requires_grad=False
# #     print(n,w)
# # print(embd)
# for k,v in weights.items():
#     v.requires_grad=False
#     print(k,v)
# import json
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/zsre-origin.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/wikidata_recent_origin.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/wikidata_counterfact_origin.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/wikibio_origin.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/zsre--aa.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/zsre_train.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/fever_train.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/fever_eval.json'
# path = '/home/zhuliyi/code/ICE_prefix_subject/data/zsre_eval.json'



# with open(path, 'r') as f:
#     result = json.load(f)
# print(len(result))
# exit()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         out = F.relu(self.fc1(x))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         return out

# features = []
# def hook(module, input, output):
#     features.append(output.clone().detach())


# net = LeNet()
# x = torch.randn(2, 3, 32, 16*5*5)
# handle = net.fc2.register_forward_hook(hook)
# y = net(x)

# print(features[0].size())
# print(y.shape)
# handle.remove()
# print('-----------')
# A = torch.tensor([[[4,7,8]],[[8,5,3]]])
# print(A.shape)
# print(A[0].shape)
# print(A[0][0].shape)
# print(A[0][0][0])
# print(A[0][0][1])
# print(A[0][0][2])


import torch

# x = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
# y = torch.tensor([[5, 6, 7], [7, 8, 9], [9, 10, 11]])
# z = torch.where(x > 5, x, y)

# print(x>5)
# print(x)
# print(y)
# print(z)

# A = torch.rand(size=(1,12,10,64))
# B = torch.rand(size=(1,12,10,64))
# print(A.shape)
# print(B.transpose(-1,-2).shape)
# print((A@B.transpose(-1,-2)).shape)

# AA = torch.ones(size=(1,1,5,5))
# BB = torch.tensor([True,True,False,False,True])
# AA[:,:,1:4,:3]=999
# print(AA)
# print(BB)
# AA = AA>1
# print(AA)
# print(AA&BB)
# print(BB.shape)

# sstr = "    Suggan Buggan River flows into the Bass Strait. What body of water does Suggan Buggan River join? Bass Strait"
# ret = sstr.find("Suggan Buggan",1)
# print(ret)
# print(f"len: {len(sstr)}")


# import torch

# # 假设张量A的形状是 (5, seq_len_a)，其中5表示有5个ids串，seq_len_a表示每个ids串的长度（可变长情况可使用填充等方式处理）
# A = torch.tensor([[1, 2, 3, 2, 3],  # 示例的第一个ids串
#                   [6, 7, 8, 9, 10],
#                   [11, 12, 13, 14, 15],
#                   [16, 17, 18, 19, 20],
#                   [21, 22, 25, 23, 24]])

# # 假设张量B的形状是 (5, seq_len_b)，其中5个对应A中的5个，seq_len_b表示每个字串对应的长度（可变长情况可使用填充等方式处理）
# B = torch.tensor([[2, 3],  # 对应A中第一个ids串里字串 [2, 3]的编码，示例要找其在A[0]中的起始位置
#                   [8, 9],
#                   [13, 14],
#                   [18, 19],
#                   [23, 24]])


# def get_start_positions(input_ids,subject_ids):
#     start_positions = []
#     for i in range(len(input_ids)):  # 遍历A中每个ids串
#         id_seq = input_ids[i]
#         target_seq = subject_ids[i]
#         target_len = len(target_seq)
#         for j in range(len(id_seq) - target_len + 1):  # 遍历可能的起始位置
#             if torch.equal(id_seq[j:j + target_len], target_seq):  # 比较切片是否和目标字串相等
#                 start_positions.append(j)
#                 break  # 找到就跳出内层循环，进入下一个ids串与对应字串的匹配
        
#     return start_positions

# print(get_start_positions(A,B))


# for i in range(7,11):
#     print(i)



