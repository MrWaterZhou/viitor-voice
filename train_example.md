## step1: Download dataset and base models
```bash
mkdir tool_models
cd tool_models
git clone https://huggingface.co/Qwen/Qwen2-0.5B
cd ..


mkdir datasets
cd datasets
mkdir emilia
wget "https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/ZH/ZH-B000000.tar" -O emilia/ZH-B000000.tar
cd ..
```

## step2: preprocess model
```bash
python viitor_voice/preprocess/preprocess_model.py tool_models/Qwen2-0.5B tool_models/Qwen2-0.5B-snac
```

## step3: preprocess dataset
```bash
# take emilia for example, if you are using other dataset, modify viitor_voice/preprocess/preprocess_model.py
python viitor_voice/preprocess/preprocess_dataset.py datasets/emilia/ZH-B000000.tar

python viitor_voice/preprocess/pack_datasets.py \
    --tokenizer_path tool_models/Qwen2-0.5B-snac \
    --data_type json \
    --data_file "datasets/emilia/*.json" \
    --max_length 2048 \
    --save_path datasets/emilia.parquet
```

## step4: start training
```
deepspeed --include localhost:0,1 train.py     \
    --model_path tool_models/Qwen2-0.5B-snac \
    --model_type decoder     \
    --data_type parquet     \
    --data_file "datasets/emilia.parquet"     \
    --streaming false     \
    --max_length 2048 \
    --merge_inputs false     \
    --output_dir checkpoints/demo     \
    --deepspeed ds1.json     \
    --per_device_train_batch_size=4     \
    --gradient_accumulation_steps=32     \
    --per_device_eval_batch_size=4     \
    --save_steps 5000 \
    --save_total_limit 3     \
    --learning_rate 1e-4     \
    --logging_steps 10     \
    --num_train_epochs 100 \
    --group_by_length false \
    --torch_compile false     \
    --do_train \
    --overwrite_output_dir \
    --bf16 \
    --remove_unused_columns false \
    --preprocessing_num_workers 24 \
    --save_safetensors false \
    --split_batches false \
    --lr_scheduler_type "cosine_with_restarts" \
    --warmup_steps 2000 \
    --lr_scheduler_kwargs '{"num_cycles": 5}'
```

