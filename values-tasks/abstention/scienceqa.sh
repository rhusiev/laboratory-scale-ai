#!/bin/sh


python finetune_science_qa.py\
    --model_id mistralai/Mistral-7B-Instruct-v0.1 \
    --save_dir mistral_sqa_final \
    --peft_save_dir mistral_sqa_peft \
    --log_dir logs \
    --run_name mistral_sqa_finetune \
    --wandb_name mistral_sqa_finetune \
    --use_model_prompt_defaults mistral \
    --hub_save_id mistral_sqa_four_bit
python finetune_science_qa.py\
    --model_id meta-llama/Llama-2-7b-chat-hf \
    --save_dir llama-2-chat_sqa_final \
    --peft_save_dir llama-2-chat_sqa_peft \
    --log_dir logs \
    --run_name llama-2-chat_sqa_finetune \
    --wandb_name llama-2-chat_sqa_finetune \
    --use_model_prompt_defaults llama-2 \
    --hub_save_id llama-2-chat_sqa_four_bit
python finetune_science_qa.py \
    --model_id tiiuae/falcon-7b-instruct \
    --save_dir falcon_sqa_final \
    --peft_save_dir falcon_sqa_peft \
    --log_dir logs \
    --run_name falcon_factcheck_sqa \
    --wandb_name falcon_sqa_finetune \
    --use_model_prompt_defaults falcon \
    --hub_save_id falcon_factcheck_four_bit

