#!/usr/bin/env python3

import logging
import sys
import transformers
import datasets
import argparse
import wandb

from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an AIME model.")

    # Model ID
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        help="The model ID to fine-tune.",
    )
    parser.add_argument(
        "--hf_token_var",
        type=str,
        default="HF_TOKEN",
        help="Name of the HuggingFace API token variable name.",
    )

    # Dataset arguments
    parser.add_argument(
        "--input_col",
        type=str,
        default="question",
        help="The name of the input column in the dataset.",
    )
    parser.add_argument(
        "--explain_col",
        type=str,
        default="answer",
        help="The name of the target column in the dataset.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="label",
        help="The name of the target column in the dataset.",
    )

    # Saving arguments
    parser.add_argument(
        "--save_model",
        type=str,
        default="True",
        help="Whether to save the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="finetuned_model",
        help="The directory to save the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--peft_save_dir",
        type=str,
        default="peft_model",
        help="The directory to save the PEFT model.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="The directory to save the results to",
        default="results",
    )

    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The directory to save the log file.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="The log level to use for fine-tuning.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="peft-aime",
        help="The name of the run, for logging.",
    )

    # W&B logging arguments
    parser.add_argument(
        "--wandb_logging", type=str, default="True", help="Whether to log to W&B."
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="peft-aime",
        help="The name of the W&B project, for logging.",
    )
    parser.add_argument(
        "--wandb_api_var",
        type=str,
        default="WANDB_API_KEY",
        help="Name of the WandB API key variable name.",
    )

    # Prompt arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="The maximum sequence length to use for fine-tuning.",
    )

    # Hub arguments
    parser.add_argument(
        "--hub_upload",
        type=str,
        default="True",
        help="Whether to upload the model to the hub.",
    )
    parser.add_argument(
        "--hub_save_id",
        type=str,
        default="rad1an/AIMELlama-3.1-8B",
        help="The name under which the model will be saved on the hub.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="The number of steps between saving the model to the hub.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/aime_22-23_think.csv",
        help="The path to the dataset.",
    )

    # Parse arguments
    args = parser.parse_args()
    model_id = args.model_id

    # HF Login
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))

    # Initialize W&B
    if args.wandb_logging == "True":
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, name=args.run_name, config=args)

    # Create directories if they do not exist
    if not path.exists(args.peft_save_dir):
        mkdir(args.peft_save_dir)
        print(f"Created directory {args.peft_save_dir}")

    if not path.exists(args.results_dir):
        mkdir(args.results_dir)
        print(f"Created directory {args.results_dir}")

    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
        print(f"Created directory {args.log_dir}")

    # Create a logger
    logger = logging.getLogger(__name__)

    # Setup logging
    print("Setting up logging...")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Use the default log level matching the training args
    log_level = args.log_level.upper()
    logger.setLevel(log_level)

    # Set the log level for the transformers and datasets libraries
    transformers.utils.logging.get_logger("transformers").setLevel(log_level)
    datasets.utils.logging.get_logger("datasets").setLevel(log_level)

    # Log to file
    file_handler = logging.FileHandler(path.join(args.log_dir, f"{args.run_name}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Get model and tokenizer
    print("Getting model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
    )

    logger.info(f"Loaded Model ID: {args.model_id}")
    dataset = datasets.load_dataset("csv", data_files=args.dataset_path, delimiter=";")
    def mapper(row):
        messages = [
            {
                "role": "system",
                "content": "You are a mathematics assistant that helps solve AIME problems. First think through the problem step by step, then when asked for the final answer, respond only with the integer number between 0 and 1000, without any explanation.",
            },
            {"role": "user", "content": row[args.input_col]},
            {"role": "assistant", "content": row[args.explain_col]},
            {"role": "user", "content": "What is the final answer?"},
            {"role": "assistant", "content": f"{row[args.target_col]}"},
        ]
        example = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": example}

    dataset = dataset.map(mapper, batched=False).remove_columns(
        [args.input_col, args.explain_col, args.target_col]
    )
    print(dataset)
    dataset.save_to_disk("acme_chat")
    logger.info("Loaded Dataset")

    # Instantiate trainer
    print("Instantiating trainer...")

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir=args.peft_save_dir,
            seed=0,
            log_level=args.log_level,
            report_to=["wandb"] if args.wandb_logging == "True" else [],
            save_steps=args.save_steps,
            push_to_hub=args.hub_upload == "True",
        ),
    )

    logger.info(f"Instantiated Trainer")

    # Fine-tune model
    print("Fine-tuning model...")

    trainer.train()

    logger.info(f"Completed fine-tuning")

    # Save adapter weights and tokenizer
    if args.save_model == "True":
        print("Saving model and tokenizer...")

        if not path.exists(args.save_dir):
            mkdir(args.save_dir)
            print(f"Created directory {args.save_dir}")

        model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

        logger.info(f"Saved model and tokenizer to {args.save_dir}")

    # Save model to hub
    if args.hub_upload == "True":
        print("Saving model to hub...")

        trainer.model.push_to_hub(args.hub_save_id, use_auth_token=True)

        logger.info(f"Saved model to hub")

    if args.wandb_logging == "True":
        wandb.finish()
