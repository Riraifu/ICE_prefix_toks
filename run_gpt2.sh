python examples/run_knowedit_llama2.py \
    --editing_method=ICE \
    --hparams_dir=./hparams/ICE/gpt2.yaml \
    --datatype='zsre' \
    --metrics_save_dir=./results/gpt2/ICE \
    --data_dir=./data/zsre_train_100000.json \

    # --data_dir=./data/zsre_eval_1000.json \
    # --data_dir=./data/zsre_train_10000.json \
# --data_dir=./data/zsre_eval_10.json \
# --data_dir=./data/zsre_train_2000_4000.json \
# --data_dir=./data/zsre_train_4000_6000.json \
# --data_dir=./data/zsre_train_6000_8000.json \
# --data_dir=./data/zsre_train_8000_10000.json \
# --data_dir=./data/zsre_train_error_examples.json \
# --data_dir=./data/zsre_train_sth_examples.json \
