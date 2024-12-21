import json

def transform_data(original_data):
    new_data = []
    for idx,item in enumerate(original_data):
        if idx>=150000:break
        new_item = {
            "subject": item["subject"],
            "prompt": item["src"],
            "target_new": item["ans"],
            "ground_truth": [item["alt"]],
            "rephrase_prompt": item["rephrase"],
            "portability": {
                "Generality": [
                    {
                        "prompt": item["rephrase"],
                        "ground_truth": item["ans"]
                    }
                ]
            },
            "locality": {
                "Locality": [
                    {
                        "prompt": item["loc"],
                        "ground_truth": [item["loc_ans"]]
                    }
                ]
            },
            "context": [
                "Knowledge may have been updated.",
                "Knowledge may have been updated.",
                "Knowledge may have been updated.",
                "Knowledge may have been updated.",
                "Knowledge may have been updated."
            ]
        }
        new_data.append(new_item)
    return new_data

# 读取原始的json文件内容
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/data/zsre_eval-001.json"
# path = "/home/zhuliyi/code/ICE_prefix_subject_GS/data/zsre_eval.json"
path = "/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train.json"

with open(path, 'r') as file:
    data = json.load(file)
# print(data)
print(len(data))
transformed_data = transform_data(data)
print(len(transformed_data))
# 将转换后的数据保存为新的json文件
output = "/home/zhuliyi/code/ICE_prefix_toks/data/zsre_train_150000.json"
with open(output, 'w') as file:
    json.dump(transformed_data, file, indent=4)