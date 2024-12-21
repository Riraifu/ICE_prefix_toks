export CUDA_VISIBLE_DEVICES=0

# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/gpt2/ICE/ICE_zsre_gpt2_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/gpt2/ICE/ICE_zsre_gpt2_results.json' 

python eval.py \
    --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
    --output_file='/home/zhuliyi/code/ICE_prefix_toks/outputs/gpt2/ICE/GPT2_attn_token_ICE_zsre_gpt2_gen_sentence.json'\
    --result_file='/home/zhuliyi/code/ICE_prefix_toks/results/gpt2/ICE/GPT2_attn_token_ICE_zsre_gpt2_results_train_10000.json' 


# python eval.py \
    # --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
    # --output_file='/home/zhuliyi/code/ICE_prefix_subject_GS/outputs/Llama-2-7b-hf/ICE/GS_LS_eval_10_ICE_zsre_Llama-2-7b-hf_gen_sentence.json'\
    # --result_file='/home/zhuliyi/code/ICE_prefix_subject_GS/results/llama-7b/ICE/GPT2_attn_token_ICE_zsre_Llama-2-7b-hf_results.json' 



# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/gpt2/ICE/ICE_zsre_gpt2_gen_sentence_prelen5.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/gpt2/ICE/ICE_zsre_gpt2_results_prelen5.json' 



# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/gpt2/ICE/ICE_zsre_gpt2_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/gpt2/ICE/ICE_zsre_gpt2_results.json' 


# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/gpt2/ICE/50_ICE_zsre_gpt2_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/gpt2/ICE/50_ICE_zsre_gpt2_results.json' 



# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/gpt2'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/gpt2/ICE/little_try_ICE_zsre_gpt2_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/gpt2/ICE/little_try_ICE_zsre_gpt2_results.json' 

# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/all_avgAlltokens_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/all_avgAlltokens_ICE_zsre_Llama-2-7b-chat-hf_results.json' 

# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/lasttoken_08_20steps_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/lasttoken_08_20steps_ICE_zsre_Llama-2-7b-chat-hf_results.json' 

# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/lasttoken_06_20steps_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/lasttoken_06_20steps_ICE_zsre_Llama-2-7b-chat-hf_results.json' 

# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/avgtokens_06_20steps_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/avgtokens_06_20steps_ICE_zsre_Llama-2-7b-chat-hf_results.json' 


# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject_GS/outputs/Llama-2-7b-chat-hf/ICE/GS_LS_1000_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject_GS/results/llama-7b/ICE/GS_LS_10000_ICE_zsre_Llama-2-7b-hf_results.json' 




# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/ICE_zsre_Llama-2-7b-chat-hf_results.json' 


# python eval.py \
#     --model_name_or_path='/home/zhuliyi/LLM_bases/Llama-2-7b-chat-hf'\
#     --output_file='/home/zhuliyi/code/ICE_prefix_subject/outputs/Llama-2-7b-chat-hf/ICE/50_ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json'\
#     --result_file='/home/zhuliyi/code/ICE_prefix_subject/results/llama-7b/ICE/50_ICE_zsre_Llama-2-7b-chat-hf_results.json' 
