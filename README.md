**Single GPU**
- for LLaMA
```
python uniform_finetune.py --model_type llama --model_name_or_path yahma/llama-7b-hf \
    --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 --learning_rate 1e-4 --epochs 10
```


**Multiple GPUs**
- for LLaMA
```
python -m torch.distributed.launch --nproc_per_node 4  \
    --nnodes=1 --node_rank=0 --master_addr=xxx --master_port=yyy uniform_finetune.py \
    --model_type llama --model_name_or_path yahma/llama-13b-hf \
    --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 --learning_rate 1e-4 --epochs 10
```

### Inference
```
python3 generate.py  --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --model_type llama

```
