# fine tuning for scienceqa task on qasper dataaset

fine_tuning script is scienceqa.sh. detailed running commands are as follows: 
```
python finetune_science_qa.py\
    --model_id mistralai/Mistral-7B-Instruct-v0.1 \
    --save_dir mistral_sqa_final \
    --peft_save_dir mistral_sqa_peft \
    --log_dir logs \
    --run_name mistral_sqa_finetune \
    --wandb_name mistral_sqa_finetune \
    --use_model_prompt_defaults mistral \
    --hub_save_id mistral_sqa_four_bit
```

# evaluation 
fine_tuning scripts are eval_scienceqa.sh and evaluate_sqa_openai.sh

```
python evaluate_sqa.py \
    --model_id /home/bingbw/lab-scale-ai/tasks/finetuned_model_mistral \
    --use_model_prompt_defaults mistral \
    --nshot "0" \
    --pretrain False \
    --wandb_name finetued_mistral_sqa_eval \
    --log_dir logs \
    --no_context
```