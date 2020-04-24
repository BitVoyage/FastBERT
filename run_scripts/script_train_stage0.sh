python3 train.py \
    --model_config_file='config/fastbert_cls.json' \
    --save_model_path='saved_model/fastbert_test' \
    --run_mode=train \
    --train_stage=0 \
    --train_data='./sample/ChnSentiCorp/train.tsv' \
    --eval_data='./sample/ChnSentiCorp/dev.tsv' \
    --epochs=8 \
    --batch_size=32 \
    --data_load_num_workers=2 \
    --gpu_ids='0,1' \
    --debug_break=0
