from collections.abc import Sequence
import numpy as np
import json
import argparse
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_from_hf_model import generate_from_prompt
from finetune import QUANZATION_MAP

from datasets import load_dataset
from tqdm import tqdm
from os import path, makedirs, getenv

import re
from typing import Optional

TEMPLATE = [
    # from aime 2018 I
    {"role": "system", "content": "After a user asks a question think about how to solve the problem. When asked to give an answer, reply with just a number. Each answer is an integer between 0 and 1000."},
    {"role": "user", "content": "Let $S$ be the number of ordered pairs of integers $(a,b)$ with $1 \\leq a \\leq 100$ and $b \\geq 0$ such that the polynomial $x^2+ax+b$ can be factored into the product of two (not necessarily distinct) linear factors with integer coefficients. Find the remainder when $S$ is divided by $1000$."},
    {"role": "assistant", "content": r"""Notice that for $x^2+ax+b$ to be true, for every $a$, $b$ will always be the product of the possibilities of how to add two integers to $a$. For example, if $a=3$, $b$ will be the product of $(3,0)$ and $(2,1)$, as those two sets are the only possibilities of adding two integers to $a$. Note that order does not matter. If we just do some simple casework, we find out that:

if $a$ is odd, there will always be $\left\lceil\frac{a}{2}\right\rceil$ $\left(\text{which is also }\frac{a+1}{2}\right)$ possibilities of adding two integers to $a$.

if $a$ is even, there will always be $\frac{a}{2}+1$ possibilities of adding two integers to $a$.

Using the casework, we have $1+2+2+3+3+...50+50+51$ possibilities. This will mean that the answer is \[\frac{(1+51)\cdot100}{2}\Rightarrow52\cdot50=2600\] possibilities.

Thus, our solution is $2600\bmod {1000}\equiv 600$."""},
    {"role": "user", "content": "What is the final answer?"},
    {"role": "assistant", "content": "600"},
]

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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: Sequence[dict[str, str]],
    question_column: str = "input",
    answer_column: str = "output",
    max_samples: int = None,
    min_new_tokens: int = 0,
    max_new_tokens: int = 50,
    remove_suffix: str = None,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a Hugging Face model on a AIME 2024 I task.
    """
    generation_kwargs = {"max_tokens": 900, start_prompt: "", end_prompt: ""}
    exact_match: list[bool] = []

    for idx in tqdm(range(min(max_samples, len(data))), desc="Evaluating AIME model"):
        question = data[idx][question_column]
        ground_truth = str(data[idx][answer_column])

        # Generate and decode the output string, removing the special tokens and any suffixes
        prompt = TEMPLATE + [{"role": "user", "content": question}]
        input_data = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        decoded = generate_from_prompt(model, tokenizer, input_data, **generation_kwargs)
        
        new_chat = list(prompt) + [{"role": "assistant", "content": decoded}, {"role": "user", "content": "What is the final answer?"}]
        input_data = tokenizer.apply_chat_template(new_chat, tokenize=False, add_generation_prompt=True)
        decoded = generate_from_prompt(model, tokenizer, input_data, **generation_kwargs)

        print(f"{ground_truth = } -> {decoded = }")

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None and remove_suffix in decoded:
            decoded = decoded.split(remove_suffix)[0]

        exact_match.append(compute_exact(decoded, ground_truth))

    return {"exact_match": np.mean(exact_match)}


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
        "--hf_model_id",
        type=str,
        help="The Huggingface model to evaluate",
        default="unsloth/Meta-Llama-3.1-8B",
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
        "--max_tokens",
        type=int,
        help="The maximum number of tokens to generate",
        default=50,
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

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize W&B
    if args.wandb_logging == "True":
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, name=args.run_name, config=args)

    # Load the test split of the dataset
    print("Loading dataset")
    # data = load_dataset(args.dataset, args.dataset_revision, split=args.split)
    data = load_dataset("csv", data_files="data/aime_2024_I.csv", delimiter=";")
    data = data["train"]

    # Model evaluation logic based on the model type
    if args.model_type == "hf":
        model_id = args.hf_model_id
        print("Loading Hugging Face model: ", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=QUANZATION_MAP['8bit'])
        model.eval()

        # Evaluate the Hugging Face model
        print("Evaluating Hugging Face model on AIME task: ", model_id)
        aime_metrics = evaluate_hf_model_aime(
            model,
            tokenizer,
            data,
            question_column="question",
            answer_column="answer",
            max_samples=args.max_samples,
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
    model_id = args.hf_model_id
    save_path = path.join(
        args.save_dir, f'{model_id.replace("/", "-")}_aime_2024_i_metrics.json'
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
