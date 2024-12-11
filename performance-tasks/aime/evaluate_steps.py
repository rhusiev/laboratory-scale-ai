from collections.abc import Sequence
import json
import argparse
import wandb

import numpy as np
import torch

import transformers
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from openai import OpenAI

from datasets import load_dataset
from tqdm import tqdm

from os import path, makedirs, getenv

import re
from typing import Optional

template = [
    {
        "role": "system",
        "content": "You are a mathematics assistant that helps solve AIME problems. First think through the problem step by step, then when asked for the final answer, respond only with the integer number between 0 and 1000, without any explanation.",
    },
]

eval_template = {
    "role": "system",
    "content": "You are a mathematics assistant that helps solve AIME problems. "
    "When asked to give a reasoning step, explain it thoroughly. "
    "When asked to validate user's reasoning step, answer with just a word 'Yes' if the reasoning step is correct. "
    "Otherwise write a correct reasoning step yourself, without replying 'No', mentioning the user's step and withoutt relating to it in any way - "
    "solve the step from scratch.",
}

#####
# TODO: Below is partially adapted better answer parsing from
# https://github.com/vlievin/medical-reasoning/blob/master/medical_reasoning/models/functional/infer_answer.py
# to be completed, and new metric added.


def parse_options_from_input(input_question: str) -> dict:
    # extract the options part from the input question
    options_str = re.search(r"\{(.+?)\}$", input_question)
    if options_str:
        options_str = options_str.group(1)
        options = dict(item.split(": ") for item in options_str.split(", "))
        return options
    else:
        return {}


def get_start_indices(target: str, pattern: str) -> list[int]:
    try:
        matches = re.finditer(pattern, target)
        return [m.start() for m in matches]
    except Exception as exc:
        return []


def get_first_match(query, choices, keys, op=min):
    assert len(choices) == len(keys)
    indices = [(key, get_start_indices(query, o)) for key, o in zip(keys, choices)]
    indices = list(filter(lambda x: len(x[1]), indices))
    if len(indices):
        return op(indices, key=lambda x: x[1])[0]
    else:
        return None


def infer_answer_from_input(input_question: str, target_answer: str) -> Optional[str]:
    options = parse_options_from_input(input_question)
    if not options:
        return None

    # check if the target answer is directly one of the option keys
    if target_answer.strip() in options:
        return target_answer.strip()

    # direct match with the provided options' values
    for key, value in options.items():
        if value.strip() == target_answer.strip():
            return key

    # use regex patterns to match the answer
    option_symbols = list(options.keys())
    option_values = list(options.values())
    option_symbols_re = [rf"{re.escape(o)}(\)|:|\.|,| )" for o in option_symbols]

    # try to match using option symbols
    match = get_first_match(target_answer, option_symbols_re, option_symbols)
    if match is not None:
        return match

    # try to match using the full text of the options
    match = get_first_match(target_answer, option_values, option_symbols)
    if match is not None:
        return match

    return None


###########
# Following code from SQUAD, here:
# https://github.com/huggingface/transformers/blob/main/src/transformers/data/metrics/squad_metrics.py


def normalize_answer(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    try:
        int_pred = int(normalize_answer(a_pred))
        int_gold = int(normalize_answer(a_gold))
    except ValueError:
        print("Non-number in answer")
        return False
    return int_gold == int_pred


############


def evaluate_hf_model_aime(
    pipeline,
    data: Sequence[dict[str, str]],
    question_column: str = "input",
    answer_column: str = "output",
    max_new_tokens: int = 4096,
    max_samples: int = None,
    remove_suffix: str = None,
    eval_each_step: bool = False,
) -> dict:
    """
    Evaluate a Hugging Face model on a AIME 2024 I task.
    """
    exact_match: list[bool] = []
    substr_match: list[bool] = []

    steps: list[list[str]] = []

    if eval_each_step:
        client = OpenAI()
        correct_steps = 0

    for idx in tqdm(range(min(max_samples, len(data))), desc="Evaluating AIME model"):
        steps.append([])
        question = data[idx][question_column]
        ground_truth = str(data[idx][answer_column])

        i = 0
        prompt = template + [
            {
                "role": "user",
                "content": f"{question}\n\n# To work this out I would first do the following step\n\n{data[idx]['step1']}\n\n# Your task\n\nComplete this step, and I will give you the next instruction.",
            }
        ]
        while True:
            print("===User===")
            print(prompt[-1]["content"].replace("\n\n", "\n"))
            decoded = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
            )[0]["generated_text"]
            last_line = decoded[-1]["content"].split("\n")[-1].strip()
            if "last step?" in last_line.lower():
                decoded[-1]["content"] = decoded[-1]["content"][
                    : max(
                        decoded[-1]["content"].rfind("."),
                        decoded[-1]["content"].rfind("\n"),
                    )
                ]
            steps[-1].append(decoded[-1])
            i += 1
            # to account for limited context size
            if i >= 5 or idx == 7 and i >= 4:
                decoded = (
                    decoded[: 2 * (i - 6) + 2]
                    + [
                        {
                            "role": "assistant",
                            "content": "I did some calculations I will use in the next step.",
                        }
                    ]
                    + decoded[2 * (i - 6) + 3 :]
                )
            if eval_each_step:
                eval_prompt = (
                    [eval_template]
                    + decoded[1:-2]
                    + [
                        {
                            "role": "user",
                            "content": f"# I think of doing the following reasoning step\n\n{data[idx][f'step{i}']}\n\n# Here is my thought process of completing the step\n\n{decoded[-1]['content']}\n\n# Your task\n\nCheck whether my reasoning step is correct.",
                        }
                    ]
                )
                response = (
                    client.chat.completions.create(
                        model="gpt-4o",
                        messages=eval_prompt,
                    )
                    .choices[0]
                    .message.content
                )
                if response.lower().startswith("yes"):
                    correct_steps += 1
                    print("===Correct reasoning step:===")
                else:
                    print("===Original reasoning step:===")
                    print(decoded[-1]["content"].replace("\n\n", "\n"))
                    decoded = decoded[:-1] + [
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    ]
                    print("===Corrected reasoning step:===")
                print(decoded[-1]["content"].replace("\n\n", "\n"))
            if f"step{i}" not in data[idx] or data[idx][f"step{i}"] in [" ", "", None]:
                break
            prompt = decoded + [
                {
                    "role": "user",
                    "content": f"# My next step would be\n\n{data[idx][f'step{i}']}\n\n# Your task\n\nDo this step, and I will give you the next instruction.",
                }
            ]

        prompt = decoded + [
            {
                "role": "user",
                "content": "# These were all the steps I had\n\n## Think about the final answer.",
            }
        ]
        print("===User===")
        print(prompt[-1]["content"].replace("\n\n", "\n"))
        decoded = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
        )[0]["generated_text"]
        if eval_each_step:
            eval_prompt = (
                [eval_template]
                + decoded[1:-2]
                + [
                    {
                        "role": "user",
                        "content": f"# I think of doing the following last reasoning step thought process\n\n{decoded[-1]['content']}\n\n# Your task\n\nCheck whether my last reasoning step is correct. If it is not, provide your reasoning step without any relation to mine. Otherwise reply with 'Yes'",
                    }
                ]
            )
            response = (
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=eval_prompt,
                )
                .choices[0]
                .message.content
            )
            if response.lower() == "yes":
                correct_steps += 1
                print("===Correct last reasoning step:===")
            else:
                print("===Original last reasoning step:===")
                print(decoded[-1]["content"].replace("\n\n", "\n"))
                decoded = decoded[:-1] + [
                    {
                        "role": "assistant",
                        "content": response,
                    }
                ]
                print("===Corrected last reasoning step:===")
            print(decoded[-1]["content"].replace("\n\n", "\n"))
        prompt = decoded + [
            {
                "role": "user",
                "content": "What is the final answer? Give me a single number.",
            }
        ]
        print("===User===")
        print(prompt[-1]["content"])
        decoded = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
        )[0]["generated_text"][-1]["content"]

        print(f"{ground_truth = } -> {decoded = }")

        exact_match.append(compute_exact(decoded, ground_truth))
        substr_match.append(normalize_answer(ground_truth) in normalize_answer(decoded))

    with open("steps.json", "w") as f:
        json.dump(steps, f)
    return {
        "exact_match": np.mean(exact_match),
        "substr_match": np.mean(substr_match),
        "correct_steps": correct_steps if eval_each_step else 0,
        "steps_count": sum(len(s) for s in steps) if eval_each_step else 0,
    }


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a AIME 2024 I task."
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of model to evaluate (currently only Huggingface)",
        default="hf",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="The Huggingface model to evaluate",
        default="unsloth/llama-3-8b-Instruct-bnb-4bit",
    )

    # Dataset arguments
    parser.add_argument(
        "--max_samples",
        type=int,
        help="The maximum number of samples to evaluate",
        default=100,
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="The maximum number of tokens to generate",
        default=4096,
    )
    parser.add_argument(
        "--remove_suffix",
        type=str,
        help="The suffix to remove from the generated output",
        default=None,
    )

    # Environment and reproducibility arguments
    parser.add_argument(
        "--device", type=str, help="The device to use for inference", default="cuda"
    )
    parser.add_argument("--seed", type=int, help="The random seed to use", default=42)
    parser.add_argument(
        "--save_dir",
        type=str,
        help="The directory to save the results to",
        default="results",
    )

    # W&B logging arguments
    parser.add_argument(
        "--wandb_logging", type=str, default="False", help="Whether to log to W&B."
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="aime_2024_i_eval",
        help="The name of the W&B project, for logging.",
    )
    parser.add_argument(
        "--wandb_api_var",
        type=str,
        default="WANDB_API_KEY",
        help="Name of the WandB API key variable name.",
    )
    parser.add_argument(
        "--eval_each_step",
        type=str,
        default="no",
        help="Whether to evaluate each step separately.",
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.eval_each_step == "yes":
        eval_each_step = True
    else:
        eval_each_step = False

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize W&B
    if args.wandb_logging == "True":
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, name=args.run_name, config=args)

    # Load the test split of the dataset
    print("Loading dataset")
    # data = load_dataset(args.dataset, args.dataset_revision, split=args.split)
    data = load_dataset("csv", data_files="data/aime_2024_I_steps.csv", delimiter=";")
    data = data["train"]

    # Model evaluation logic based on the model type
    if args.model_type == "hf":
        model_id = args.model_id
        print("Loading Hugging Face model: ", model_id)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # Evaluate the Hugging Face model
        print("Evaluating Hugging Face model on AIME task: ", model_id)
        aime_metrics = evaluate_hf_model_aime(
            pipeline,
            data,
            question_column="question",
            answer_column="answer",
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            eval_each_step=eval_each_step,
        )
    elif args.model_type == "unsloth":
        model_id = args.model_id
        print("Loading Hugging Face model: ", model_id)
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_id,
            dtype=None,  # autodetect
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/llama-3-8b-Instruct-bnb-4bit"
        )  # hardcode
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3",
            # mapping={"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
        )
        FastLanguageModel.for_inference(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device_map="auto",
        )

        # Evaluate the Hugging Face model
        print("Evaluating Hugging Face model on AIME task: ", model_id)
        aime_metrics = evaluate_hf_model_aime(
            pipeline,
            data,
            question_column="question",
            answer_column="answer",
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            eval_each_step=eval_each_step,
        )
    else:
        raise ValueError("Invalid model type: ", args.model_type)

    # Print the metrics to the console
    print("Model AIME Metrics:")
    for key, value in aime_metrics.items():
        print(f"{key}: {value}")

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **aime_metrics}

    # Save the metrics to a JSON file
    model_id = args.model_id
    save_path = path.join(
        args.save_dir, f'{model_id.replace("/", "-")}_aime_2024_i_steps_metrics.json'
    )
    print("Saving AIME metrics to: ", save_path)

    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(save_path, "w") as f:
        json.dump(metrics, f)

    # Log the metrics to W&B
    if args.wandb_logging == "True":
        wandb.log(metrics)
        wandb.finish()
