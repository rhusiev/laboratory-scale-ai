#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

python evaluate_sqa.py \
    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_mistral \
    --use_model_prompt_defaults mistral \
    --nshot "all" \
    --pretrain False \
    --wandb_name finetued_mistral_sqa_eval \
    --log_dir logs \


python evaluate_sqa.py \
    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_mistral \
    --use_model_prompt_defaults mistral \
    --nshot "0" \
    --pretrain False \
    --wandb_name finetued_mistral_sqa_eval \
    --log_dir logs \
    --no_context

python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_bllama-2 \
    --use_model_prompt_defaults finetuned_model_bllama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \
    --no_context

python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_bllama-2 \
    --use_model_prompt_defaults finetuned_model_bllama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \


python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_b_halfllama-2 \
    --use_model_prompt_defaults finetuned_model_b_halfllama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \
    --no_context

python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_b_halfllama-2 \
    --use_model_prompt_defaults finetuned_model_b_halfllama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \


python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_b_answerablellama-2 \
    --use_model_prompt_defaults finetuned_model_b_answerablellama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \
    --no_context

python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_b_answerablellama-2 \
    --use_model_prompt_defaults finetuned_model_b_answerablellama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \


python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_llama-2 \
    --use_model_prompt_defaults finetuned_model_llama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \
    --no_context

python evaluate_sqa.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --model_id /home/bingbw/fine_tuning/finetuned_model_llama-2 \
    --use_model_prompt_defaults finetuned_model_llama-2 \
    --nshot "0" \
    --wandb_name finetued_llama_sqa_eval \
    --log_dir logs \
    --pretrain False \




#
#
#python evaluate_sqa.py \
#    --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#    --use_model_prompt_defaults mistral \
#    --wandb_name Mistral_sqa_eval \
#    --log_dir logs \
#    --nshot "0" \
#    --no_context

#


#python evaluate_sqa.py \
#    --model_id meta-llama/Llama-2-13b-chat-hf \
#    --use_model_prompt_defaults llama-2 \
#    --wandb_name Llama_sqa_eval \
#    --log_dir logs \
#    --nshot "0" \
#
#python evaluate_sqa.py \
#    --model_id meta-llama/Llama-2-13b-chat-hf \
#    --use_model_prompt_defaults llama-2 \
#    --wandb_name Llama_sqa_eval \
#    --log_dir logs \
#    --nshot "0" \
#    --no_context
#




#python evaluate_sqa.py \
#    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_falcon \
#    --use_model_prompt_defaults falcon \
#    --nshot "0" \
#    --wandb_name finetued_falcon_sqa_eval \
#    --log_dir logs \
#    --pretrain False \
#    --no_context
#
#python evaluate_sqa.py \
#    --model_id tiiuae/falcon-7b-instruct \
#    --use_model_prompt_defaults falcon \
#    --wandb_name falcon_sqa_eval \
#    --log_dir logs \
#    --nshot "0" \
#    --no_context





#python evaluate_sqa.py \
#    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_mistral \
#    --use_model_prompt_defaults mistral \
#    --nshot all \
#    --pretrain False \
#    --wandb_name finetued_mistral_sqa_eval \
#    --log_dir logs \
#
#
#python evaluate_sqa.py \
#    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_llama-2 \
#    --use_model_prompt_defaults llama-2 \
#    --nshot all \
#    --wandb_name finetued_llama_sqa_eval \
#    --log_dir logs \
#    --pretrain False
#
#python evaluate_sqa.py \
#    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_falcon \
#    --use_model_prompt_defaults falcon \
#    --nshot all \
#    --wandb_name finetued_falcon_sqa_eval \
#    --log_dir logs \
#    --pretrain False
#
#python evaluate_sqa.py \
#    --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#    --use_model_prompt_defaults mistral \
#    --wandb_name Mistral_sqa_eval \
#    --log_dir logs \
#    --nshot all
#
#python evaluate_sqa.py \
#    --model_id meta-llama/Llama-2-7b-chat-hf \
#    --use_model_prompt_defaults llama-2 \
#    --wandb_name Llama_sqa_eval \
#    --log_dir logs \
#    --nshot all
#
#python evaluate_sqa.py \
#    --model_id tiiuae/falcon-7b-instruct \
#    --use_model_prompt_defaults falcon \
#    --wandb_name falcon_sqa_eval \
#    --log_dir logs \
#    --nshot all



#python finetune_science_qa.py\
#    --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#    --save_dir mistral_sqa_final \
#    --peft_save_dir mistral_sqa_peft \
#    --log_dir logs \
#    --run_name mistral_sqa_finetune \
#    --wandb_name mistral_sqa_finetune \
#    --use_model_prompt_defaults mistral \
#    --hub_save_id mistral_sqa_four_bit
#python finetune_science_qa.py\
#    --model_id meta-llama/Llama-2-7b-chat-hf \
#    --save_dir llama-2-chat_sqa_final \
#    --peft_save_dir llama-2-chat_sqa_peft \
#    --log_dir logs \
#    --run_name llama-2-chat_sqa_finetune \
#    --wandb_name llama-2-chat_sqa_finetune \
#    --use_model_prompt_defaults llama-2 \
#    --hub_save_id llama-2-chat_sqa_four_bit
#python finetune_science_qa.py \
#    --model_id tiiuae/falcon-7b-instruct \
#    --save_dir falcon_sqa_final \
#    --peft_save_dir falcon_sqa_peft \
#    --log_dir logs \
#    --run_name falcon_factcheck_sqa \
#    --wandb_name falcon_sqa_finetune \
#    --use_model_prompt_defaults falcon \
#    --hub_save_id falcon_factcheck_four_bit

#
#if [ "$modelID" = "all" ] && [ "$taskID" = "evaluate-full" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 3
#fi
#
#if [ "$modelID" = "all" ] && [ "$taskID" = "all" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 3
#    python finetune_factcheck.py \
#        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --save_dir mistral_factcheck_final \
#        --peft_save_dir mistral_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name mistral_factcheck_finetune \
#        --wandb_name mistral_factcheck_finetune \
#        --use_model_prompt_defaults mistral \
#        --hub_save_id mistral_factcheck_four_bit
#    python finetune_factcheck.py \
#        --model_id meta-llama/Llama-2-7b-chat-hf \
#        --save_dir llama-2-chat_factcheck_final \
#        --peft_save_dir llama-2-chat_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name llama-2-chat_factcheck_finetune \
#        --wandb_name llama-2-chat_factcheck_finetune \
#        --use_model_prompt_defaults llama-2 \
#        --hub_save_id llama-2-chat_factcheck_four_bit
#    python finetune_factcheck.py \
#        --model_id tiiuae/falcon-7b-instruct \
#        --save_dir falcon_factcheck_final \
#        --peft_save_dir falcon_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name falcon_factcheck_finetune \
#        --wandb_name falcon_factcheck_finetune \
#        --use_model_prompt_defaults falcon \
#        --hub_save_id falcon_factcheck_four_bit
#fi
#
#if [ "$modelID" = "all" ] && [ "$taskID" = "test" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#    python finetune_factcheck.py \
#        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --save_dir mistral_factcheck_final \
#        --peft_save_dir mistral_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name mistral_factcheck_finetune \
#        --wandb_name mistral_factcheck_finetune \
#        --use_model_prompt_defaults mistral \
#        --hub_save_id mistral_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#    python finetune_factcheck.py \
#        --model_id meta-llama/Llama-2-7b-chat-hf \
#        --save_dir llama-2-chat_factcheck_final \
#        --peft_save_dir llama-2-chat_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name llama-2-chat_factcheck_finetune \
#        --wandb_name llama-2-chat_factcheck_finetune \
#        --use_model_prompt_defaults llama-2 \
#        --hub_save_id llama-2-chat_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#    python finetune_factcheck.py \
#        --model_id tiiuae/falcon-7b-instruct \
#        --save_dir falcon_factcheck_final \
#        --peft_save_dir falcon_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name falcon_factcheck_finetune \
#        --wandb_name falcon_factcheck_finetune \
#        --use_model_prompt_defaults falcon \
#        --hub_save_id falcon_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#fi
#
#if [ "$modelID" = "all" ] && [ "$taskID" = "test-eval" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#    	--split test[:50] \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#    	--split test[:50] \
#        --shots 3
#fi
#
#if [ "$modelID" = "all" ] && [ "$taskID" = "test-finetune" ]; then
#    python finetune_factcheck.py \
#        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --save_dir mistral_factcheck_final \
#        --peft_save_dir mistral_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name mistral_factcheck_finetune \
#        --wandb_name mistral_factcheck_finetune \
#        --use_model_prompt_defaults mistral \
#        --hub_save_id mistral_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#    python finetune_factcheck.py \
#        --model_id meta-llama/Llama-2-7b-chat-hf \
#        --save_dir llama-2-chat_factcheck_final \
#        --peft_save_dir llama-2-chat_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name llama-2-chat_factcheck_finetune \
#        --wandb_name llama-2-chat_factcheck_finetune \
#        --use_model_prompt_defaults llama-2 \
#        --hub_save_id llama-2-chat_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#    python finetune_factcheck.py \
#        --model_id tiiuae/falcon-7b-instruct \
#        --save_dir falcon_factcheck_final \
#        --peft_save_dir falcon_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name falcon_factcheck_finetune \
#        --wandb_name falcon_factcheck_finetune \
#        --use_model_prompt_defaults falcon \
#        --hub_save_id falcon_factcheck_four_bit-test \
#        --train_slice train[:50] \
#        --test_slice test[:50]
#fi
#
#if [ "$modelID" = "mistral" ] && [ "$taskID" = "all" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --use_model_prompt_defaults mistral \
#        --results_dir results \
#        --run_name mistral_factcheck_eval \
#        --wandb_name mistral_factcheck_eval \
#        --shots 3
#    python finetune_factcheck.py \
#        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
#        --save_dir mistral_factcheck_final \
#        --peft_save_dir mistral_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name mistral_factcheck_finetune \
#        --wandb_name mistral_factcheck_finetune \
#        --use_model_prompt_defaults mistral \
#        --hub_save_id mistral_factcheck_four_bit
#fi
#
#if [ "$modelID" = "llama" ] && [ "$taskID" = "all" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
#        --use_model_prompt_defaults llama-2 \
#        --results_dir results \
#        --run_name llama-2-chat_factcheck_eval \
#        --wandb_name llama-2-chat_factcheck_eval \
#        --shots 3
#    python finetune_factcheck.py \
#        --model_id meta-llama/Llama-2-7b-chat-hf \
#        --save_dir llama-2-chat_factcheck_final \
#        --peft_save_dir llama-2-chat_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name llama-2-chat_factcheck_finetune \
#        --wandb_name llama-2-chat_factcheck_finetune \
#        --use_model_prompt_defaults llama-2 \
#        --hub_save_id llama-2-chat_factcheck_four_bit
#fi
#
#if [ "$modelID" = "falcon" ] && [ "$taskID" = "all" ]; then
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 0
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 1
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 2
#    python evaluate_factcheck.py \
#        --model_type hf \
#        --hf_model_id tiiuae/falcon-7b-instruct \
#        --use_model_prompt_defaults falcon \
#        --results_dir results \
#        --run_name falcon_factcheck_eval \
#        --wandb_name falcon_factcheck_eval \
#        --shots 3
#    python finetune_factcheck.py \
#        --model_id tiiuae/falcon-7b-instruct \
#        --save_dir falcon_factcheck_final \
#        --peft_save_dir falcon_factcheck_peft \
#        --results_dir results \
#        --log_dir logs \
#        --run_name falcon_factcheck_finetune \
#        --wandb_name falcon_factcheck_finetune \
#        --use_model_prompt_defaults falcon \
#        --hub_save_id falcon_factcheck_four_bit
#fi