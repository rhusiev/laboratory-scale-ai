#!/usr/bin/env python3

import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb

from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
from typing import Mapping
from tqdm import tqdm

from finetune import (
    get_model_and_tokenizer,
    get_lora_model,
    get_default_trainer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an AIME model.")

    # Model ID
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The model ID to fine-tune.",
    )
    parser.add_argument(
        "--hf_token_var",
        type=str,
        default="HF_TOKEN",
        help="Name of the HuggingFace API token variable name.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="False",
        help="Whether to resume from a checkpoint.",
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to mount the model on."
    )
    parser.add_argument(
        "--use_mps_device",
        type=str,
        default="False",
        help="Whether to use an MPS device.",
    )
    parser.add_argument(
        "--max_memory",
        type=str,
        default="12000MB",
        help="The maximum memory per GPU, in MB.",
    )

    # Model arguments
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        default="True",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="4bit",
        help="The quantization type to use for fine-tuning.",
    )
    parser.add_argument("--lora", type=str, default="True", help="Whether to use LoRA.")
    parser.add_argument(
        "--tune_modules",
        type=str,
        default="linear4bit",
        help="The modules to tune using LoRA.",
    )
    parser.add_argument(
        "--exclude_names",
        type=str,
        default="lm_head",
        help="The names of the modules to exclude from tuning.",
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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="The maximum number of steps to use for fine-tuning.",
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
        "--logging_first_step",
        type=str,
        default="True",
        help="Whether to log the first step.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="The number of steps between logging.",
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
        default=974,
        help="The maximum sequence length to use for fine-tuning.",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for fine-tuning.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="The number of gradient accumulation steps to use for fine-tuning.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="The learning rate to use for fine-tuning.",
    )
    parser.add_argument("--fp16", type=str, default="True", help="Whether to use fp16.")
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="The optimizer to use for fine-tuning.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="The number of warmup steps to use for fine-tuning.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="The evaluation strategy to use for fine-tuning.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=250,
        help="The number of steps between evaluations.",
    )
    parser.add_argument(
        "--eval_on_test",
        type=str,
        default="True",
        help="Whether to evaluate the model on the test set after fine-tuning.",
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
        default="rad1an/peft-aime",
        help="The name under which the model will be saved on the hub.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="The number of steps between saving the model to the hub.",
    )

    # Parse arguments
    args = parser.parse_args()

    def data_formatter(
        data: Mapping,
        input_field: str = args.input_col,
        explain_field: str = args.explain_col,
        target_field: str = args.target_col,
    ) -> list[str]:
        new_data = []
        for i in tqdm(range(len(data[input_field])), desc="Formatting data"):
            messages = [
                {
                    "role": "system",
                    "content": "After a user asks a question think about how to solve the problem. When asked to give a final answer, reply with a single number, without explanations. Each answer is an integer between 0 and 1000.",
                },
                {
                    "role": "user",
                    "content": data[input_field][i],
                },
                {
                    "role": "assistant",
                    "content": data[explain_field][i],
                },
            ]
            new_data.append(messages)
            messages = [
                {
                    "role": "system",
                    "content": "After a user asks a question think about how to solve the problem. When asked to give a final answer, reply with a single number, without explanations. Each answer is an integer between 0 and 1000.",
                },
                {
                    "role": "user",
                    "content": data[input_field][i],
                },
                {
                    "role": "assistant",
                    "content": data[explain_field][i],
                },
                {
                    "role": "user",
                    "content": "What is the final answer?",
                },
                {
                    "role": "assistant",
                    "content": data[target_field][i],
                },
            ]
            new_data.append(messages)
        return new_data

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

    model, tokenizer = get_model_and_tokenizer(
        args.model_id,
        quantization_type=args.quantization_type,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        device=args.device,
    )

    tokenizer.padding_side = "right"

    logger.info(f"Loaded Model ID: {args.model_id}")

    # Get LoRA model
    if args.lora == "True":
        print("Getting LoRA model...")

        if args.tune_modules == "linear":
            lora_modules = [torch.nn.Linear]
        elif args.tune_modules == "linear4bit":
            lora_modules = [bnb.nn.Linear4bit]
        elif args.tune_modules == "linear8bit":
            lora_modules = [bnb.nn.Linear8bit]
        else:
            raise ValueError(
                f"Invalid tune_modules argument: {args.tune_modules}, must be linear, linear4bit, or linear8bit"
            )

        model = get_lora_model(
            model,
            include_modules=lora_modules,
            exclude_names=args.exclude_names,
            matrix_rank=32,
        )

        logger.info(f"Loaded LoRA Model")

    # Download and prepare data
    print("Downloading and preparing data...")

    data = datasets.load_dataset(
        "csv", data_files="data/aime_2024_I.csv", delimiter=";"
    )
    data = data["train"]

    # Get dataset splits
    train_data = data[: len(data) * 8 // 10]
    validation_data = data[len(data) * 8 // 10 :]

    # Set the format of the data
    train_data.set_format(type="torch", device=args.device)
    validation_data.set_format(type="torch", device=args.device)

    logger.info("Loaded Dataset")

    # Handle no max steps by training one dataset epoch
    if args.max_steps is None:
        args.max_steps = len(train_data)

    # Instantiate trainer
    print("Instantiating trainer...")

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16 == "True",
        logging_steps=args.logging_steps,
        output_dir=args.peft_save_dir,
        optim=args.optim,
        use_mps_device=args.use_mps_device == "True",
        log_level=args.log_level,
        logging_first_step=args.logging_first_step == "True",
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        resume_from_checkpoint=args.resume_from_checkpoint == "True",
        push_to_hub=args.hub_upload == "True",
        save_steps=args.save_steps,
        report_to=["wandb"] if args.wandb_logging == "True" else [],
    )

    trainer = get_default_trainer(
        model,
        tokenizer,
        train_data,
        eval_dataset=validation_data,
        formatting_func=data_formatter,
        max_seq_length=args.max_seq_length,
        training_args=training_args,
    )

    model.config.use_cache = False

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

        trainer.model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        logger.info(f"Saved model and tokenizer to {args.save_dir}")

    # Save model to hub
    if args.hub_upload == "True":
        print("Saving model to hub...")

        trainer.model.push_to_hub(args.hub_save_id, use_auth_token=True)

        logger.info(f"Saved model to hub")

    if args.wandb_logging == "True":
        wandb.finish()
