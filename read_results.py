path = 'results/gpt2/ICE/GPT2_attn_token_ICE_zsre_gpt2_results_train_10000.json'

import json

with open(path, 'r') as file:
    results = json.load(file)

error_examples = []
sth_examples = []
for res in results:
    ES = res['post']['rewrite_acc'][0]
    case_id = res['case_id']
    if ES==0.0:
        error_examples.append(case_id)
        # error_examples.append([case_id,ES])
    elif ES>0 and ES<1.0:
        sth_examples.append(case_id)
        # sth_examples.append([case_id,ES])

print(f"error_examples: {error_examples}")
print('===============')
print(f"sth_examples: {sth_examples}")
print('####################')
print(f"error_examples: {len(error_examples)}")
print(f"sth_examples: {len(sth_examples)}")



data_path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_10000.json'
with open(data_path, 'r') as file:
    datas = json.load(file)

print(len(datas))
error_datas = []
sth_datas = []

for idx,data in enumerate(datas):
    if idx in error_examples:
        error_datas.append(data)
    elif idx in sth_examples:
        sth_datas.append(data)
print(len(error_datas))
print(len(sth_datas))

error_data_path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_error_examples.json'
sth_data_path = '/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_sth_examples.json'


with open(error_data_path, 'w') as file:
    json.dump(error_datas, file, indent=4)
    
with open(sth_data_path, 'w') as file:
    json.dump(sth_datas, file, indent=4)