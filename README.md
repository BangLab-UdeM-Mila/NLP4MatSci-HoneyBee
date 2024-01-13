# NLP4MatSci-HoneyBee
This repository contains the dataset and code for our EMNLP'23 publication: "HoneyBee: Progressive Instruction Finetuning of Large Language Models for Materials Science".  

**Single GPU**
- for LLaMA (You need to first unzip the files in peft.zip and place them under the ./peft/ path)
```
python uniform_finetune.py --model_type llama --model_name_or_path yahma/llama-7b-hf \
    --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 --learning_rate 1e-4 --epochs 10
```


**Multiple GPUs**
- for LLaMA  (You need to first unzip the files in peft.zip and place them under the ./peft/ path)
```
python -m torch.distributed.launch --nproc_per_node 4  \
    --nnodes=1 --node_rank=0 --master_addr=xxx --master_port=yyy uniform_finetune.py \
    --model_type llama --model_name_or_path yahma/llama-13b-hf \
    --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 --learning_rate 1e-4 --epochs 10
```

### Inference (for debug)
```
python generate.py  --data ./data/formatted_cot_data/train_instructions_from_chatgpt.json --model_type llama

```

### Inference (for batch prediction)
```
python predict.py --model_type llama --size 7b --data ./data/formatted_cot_data/test_xxx.json --predict_batch_size 4 --cutoff_len 2048 --lora_dir ./saved_models/llama-7b-hf/lora
```

### Instructions Data
You can find our [instructions-based data](https://zenodo.org/records/10119842) for HonyeBee training and test via Zenodo.

### QA   
If you have any questions about this code, feel free to email yu.song@umontreal.ca. I will response as soon as possible.
