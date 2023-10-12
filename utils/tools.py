import copy
import torch
from datasets import load_dataset
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from transformers import GenerationConfig
from .config import *
from .device import get_device_map
import random



def generate_prompt(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    return prompt_.format_map(data_point)


def generate_prompt_dict(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    result_ = data_point["output"]
    return {
        "origin": data_point,
        "input": prompt_.format_map(data_point),
        "output": result_
    }


def get_data_model(args):
    def _get_model_class(llm_type, model_path):
        if llm_type not in AVAILABLE_MODEL:
            llm_type = "Auto"
            return MODEL_CLASSES[llm_type], model_path
        else:
            load_path = llm_type + "_" + model_path
            return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

    data = load_dataset("json", data_files=args.data)
    print(data)

    model_class, model_path = _get_model_class(args.model_type, args.size)

    model = model_class.model.from_pretrained(model_path,
                                              load_in_8bit=False,
                                              device_map=DEVICE_MAP)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=MODEL_LORA_TARGET_MODULES[args.model_type],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return data, model, tokenizer


def get_tokenize_func(args, tokenizer):
    def tokenize(prompt):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.cutoff_len,
                           padding=False,
                           )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "labels": copy.deepcopy(result["input_ids"])
        }

    return tokenize, tokenize


def get_train_val_data(args, data, tokenizer):
    def _generate_and_tokenize_prompt(data_point):
        prompt_no_resp = generate_prompt(data_point)
        prompt_tokenize, completion_tokenize = get_tokenize_func(args, tokenizer)
        tokenized_result = prompt_tokenize(prompt_no_resp)
        source_len = len(tokenized_result['input_ids'])
        prompt_with_response = prompt_no_resp + " " + data_point["output"]
        prompt_with_response += " " + tokenizer.eos_token
        tokenized_with_response = completion_tokenize(prompt_with_response)
        tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]
        return tokenized_with_response

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(_generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(_generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(_generate_and_tokenize_prompt)
        val_data = None
    return train_data, val_data


def get_predict_data(args):
    data = load_dataset("json", data_files=args.data)
    data = data.filter(lambda x: (len(x['input']) + len(x['instruction']) + len(x['output']))<1000)
    if (args.sample_size>0):
        length = len(data)
        ratio = args.sample_size/length
        data = data.filter(lambda x: random.random()<=ratio)
    print(data)
    predict_data = data["train"].shuffle().map(generate_prompt_dict)
    return predict_data


def get_fine_tuned_model(args):
    def _get_model_class(llm_type, model_path):
        if llm_type not in AVAILABLE_MODEL:
            llm_type = "Auto"
            return MODEL_CLASSES[llm_type], model_path
        else:
            load_path = llm_type + "_" + model_path
            if llm_type in ['moss']:
                load_path = llm_type
            return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

    model_class, model_path = _get_model_class(args.model_type, args.size)
    if args.model_type == "moss":
        model = model_class.model.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map= get_device_map(model_type="moss", load_in_8bit=True))

        tokenizer = model_class.tokenizer.from_pretrained(model_path,trust_remote_code=True)
        if args.lora_dir != 'none':
            model = PeftModel.from_pretrained(
                model,
                args.lora_dir,
                device_map={"": DEVICE_TYPE}
            )
    else:
        model = model_class.model.from_pretrained(model_path,
                                                  load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map=DEVICE_MAP)

        tokenizer = model_class.tokenizer.from_pretrained(model_path)
        if args.lora_dir != 'none':
            model = PeftModel.from_pretrained(
                model,
                args.lora_dir,
                device_map={"": DEVICE_TYPE}
            )
    model.half()
    return model, tokenizer


def get_lora_model(args):
    def _get_model_class(llm_type, model_path):
        if llm_type not in AVAILABLE_MODEL:
            llm_type = "Auto"
            return MODEL_CLASSES[llm_type], model_path
        else:
            load_path = llm_type + "_" + model_path
            return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

    model_class, model_path = _get_model_class(args.model_type, args.size)

    model = model_class.model.from_pretrained(model_path,
                                              load_in_8bit=False,
                                              torch_dtype=torch.float16,
                                              device_map={"": "cpu"}, )
    if args.lora_dir != 'none':
        lora_model = PeftModel.from_pretrained(
            model,
            args.lora_dir,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )
    else:
        lora_model = None

    if 'q_proj' in MODEL_LORA_TARGET_MODULES[args.model_type] and 'v_proj' in MODEL_LORA_TARGET_MODULES[args.model_type]:
        lora_type = 'q_v_proj'
    elif 'query_key_value' in MODEL_LORA_TARGET_MODULES[args.model_type]:
        lora_type = 'query_key_value'
    else:
        lora_type = None
    return model, lora_model, lora_type, model_class


def generate_service_prompt(instruction, llm, lora):
    if llm in ['moss']:
        return META_INSTRUCTION.get('moss',"") + PROMPT_DICT['prompt_format_before'] + instruction + PROMPT_DICT['prompt_format_after']
    return PROMPT_DICT['prompt_format_before'] + instruction + PROMPT_DICT['prompt_format_after']


def get_generation_config(llm):
    generation_configs = GenerationConfig(
        temperature=GENERATE_CONFIG['temperature'],
        top_p=GENERATE_CONFIG['top_p'],
        top_k=GENERATE_CONFIG['top_k'],
        num_beams=GENERATE_CONFIG['num_beams'],
        max_new_tokens=GENERATE_CONFIG['max_new_tokens']
    )
    return generation_configs


def generate_service_output(output, prompt, llm, lora):
    if lora == 'none':
        if llm in ['llama']:
            return output.replace(prompt, '', 1).strip()
        else:
            return output.split("### Response:")[1].strip()
    else:
        return output.split("### Response:")[1].strip()


