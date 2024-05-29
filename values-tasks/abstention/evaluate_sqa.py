import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune_functions import get_model_and_tokenizer
from evaluate_functions import evaluate_hf_model
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
import transformers
import torch
import random
import wandb
import logging
import json
import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
from typing import Mapping



MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>'
}

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-2-13b-chat-hf ')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-13b-chat-hf ')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='mistral', help='Whether to use the default prompts for a model')
    parser.add_argument('--nshot', type=str, default='all', help='The slice of the test dataset to use for fine-tuning.')
    parser.add_argument('--pretrain', type=str, default='True', help='The slice of the test dataset to use for fine-tuning.')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument("--no_context",action="store_true", help="If given, we're evaluating a model without the gold context passage.")

    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')
    parser.add_argument('--log_level', type=str, default='info', help='The log level to use for fine-tuning.')


    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='peft_finetune', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    args = parser.parse_args()
    args.run_name = 'eval_model_'+args.use_model_prompt_defaults+ str(args.no_context)

    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
        print(f'Created directory {args.log_dir}')

    if args.use_model_prompt_defaults:
        args.suffix = MODEL_SUFFIXES["llama-2"]

    # Create a logger
    logger = logging.getLogger(__name__)

    # Setup logging
    print('Setting up logging...')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Use the default log level matching the training args
    log_level = args.log_level.upper()
    logger.setLevel(log_level)


    # Log to file
    file_handler = logging.FileHandler(path.join(args.log_dir, f'{args.run_name}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    #-------------------
    # load data
    #-------------------
    print('Getting data...')
    base_url = "/home/bingbw/fine_tuning/data/"
    data = load_dataset('json', data_files={'train': base_url + 'qasper_train.json', 'test': base_url + 'qasper_test.json'})
    train_data = data['train']
    test_data = data['test']

    #---------------------------
    # prepara prompting examples
    #---------------------------
    system_message = """You are a helpful assistant!"""
    transaction = ""

    #-------------------
    # load model
    #-------------------
    logger.info(f'Loaded Model ID: {args.model_id}')
    model, tokenizer = get_model_and_tokenizer(args.base_model,
                                                   gradient_checkpointing=False,
                                                   quantization_type='4bit',
                                                   device=args.device)
    if args.pretrain == 'False':
        lora_model = PeftModel.from_pretrained(model, args.model_id)
        model = lora_model.merge_and_unload()

    #--------------
    # inference
    #--------------
    #for nshot in ['zero', 'one', 'two', 'three']:
    model.eval()
    if args.nshot == "all":
        for nshot in range(4):
            logger.info(f'eval: {str(nshot)}')
            examples_demo = ""
            if nshot > 0:
                fewshots = random.sample(range(len(train_data['context'])), nshot)
                for idx, each in enumerate(fewshots):
                    examples_demo += f"""\n\nExample {idx+1}:\n\n## Documents:\n{train_data['context'][each]}\n\n## Question:\n{train_data['question'][each]}\n\n## Answer:\n{train_data['answer'][each]}\n\n"""

            model_outputs, metrics = evaluate_hf_model(model=model,
                                                       tokenizer=tokenizer,
                                                       data=test_data,
                                                       args=args,
                                                       max_samples=len(test_data['context']),
                                                       system_message=system_message,
                                                       transaction=transaction,
                                                       examples = examples_demo,
                                                       remove_suffix=args.suffix,
                                                       shot = nshot)

            logger.info(f'eval done: {str(nshot)}')
            print(f'{nshot} Results:')
            if args.pretrain == 'True':
                for k, v in metrics.items():print(f'{k}: {v}')
                with open(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_pretrained_model_{nshot}shot_outputs_{args.no_context}.json", 'w') as f: json.dump(metrics, f)
            else:
                for k, v in metrics.items():print(f'{k}: {v}')
                with open(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_finetuned_model_{nshot}shot_outputs_{args.no_context}.json", 'w') as f: json.dump(metrics, f)


    else:
        print(f"{args.nshot}shot...")
        examples_demo = ""
        if int(args.nshot) > 0:
            fewshots = random.sample(range(len(train_data['context'])), int(args.nshot))
            for idx, each in enumerate(fewshots):
                examples_demo += f"""\n\nExample {idx + 1}:\n\n## Documents:\n{train_data['context'][each]}\n\n## Question:\n{train_data['question'][each]}\n\n## Answer:\n{train_data['answer'][each]}\n\n"""

        model_outputs, metrics = evaluate_hf_model(model=model,
                                                   tokenizer=tokenizer,
                                                   data=test_data,
                                                   args=args,
                                                   max_samples=len(test_data['context']),
                                                   system_message=system_message,
                                                   transaction=transaction,
                                                   examples = examples_demo,
                                                   remove_suffix=args.suffix,
                                                   shot = args.nshot)
        print(f'{args.nshot} Results:')
        if args.pretrain == 'True':
            for k, v in metrics.items():print(f'{k}: {v}')
            with open(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_pretrained_model_{args.nshot}shot_outputs_{args.no_context}.json", 'w') as f: json.dump(metrics, f)
        else:
            for k, v in metrics.items():print(f'{k}: {v}')
            with open(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_finetuned_model_{args.nshot}shot_outputs_{args.no_context}.json", 'w') as f: json.dump(metrics, f)


if __name__ == "__main__":
    main()
