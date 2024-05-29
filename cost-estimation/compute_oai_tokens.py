#!/usr/bin/env python3

import argparse
import torch
import tiktoken

from datasets import load_dataset
from typing import Iterable

from openai_chat_api import DialogueBot

# Default instructions and k-shot prompts for the fact-checking task and the Climate-FEVER dataset
FACT_CHECK_INSTRUCTION = 'Please classify the following Claim into one of three labels based on the Evidence that follows it. The labels are SUPPORTS, if the Claim is supported by the Evidence; REFUTES, if the Claim is refuted by the Evidence; or NOT_ENOUGH_INFO, if the Claim is neither supported nor refuted by the Evidence. Output only one word: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO. '

FACT_CHECK_ONE_SHOT = '\nHere is an example of how to perform the task. Claim: The Review concluded that CRU\'s actions were normal and did not threaten the integrity of peer review. Evidence: Describing its report as "hugely positive", he stated that "it is especially important that, despite a deluge of allegations and smears against the CRU, this independent group of utterly reputable scientists have concluded that there was no evidence of any scientific malpractice." What is the correct label? SUPPORTS\n'
FACT_CHECK_TWO_SHOT = 'Here is another example of how to perform the task. Claim: Recent research also indicates that the quantity of fossil fuels staying in the atmosphere is much less than previously thought. Evidence: (2007) concluded that unless energy policies changed substantially, the world would continue to depend on fossil fuels until 2025–2030. What is the correct label? NOT_ENOUGH_INFO\n'
FACT_CHECK_THREE_SHOT = 'Here is another example of how to perform the task. Claim: "Global warming" and "climate change" mean different things and have both been used for decades. Evidence: Global warming and climate change are often used interchangeably. What is the correct label? REFUTES\n'

FACT_CHECK_TRANSITION = 'Here is the example that needs to be classified. Please respond with only one word after being asked for the correct label. '

MULTI_TURN_DICT = {
    'question_1': 'The Review concluded that CRU\'s actions were normal and did not threaten the integrity of peer review. Evidence: Describing its report as "hugely positive", he stated that "it is especially important that, despite a deluge of allegations and smears against the CRU, this independent group of utterly reputable scientists have concluded that there was no evidence of any scientific malpractice."',
    'question_2': 'Recent research also indicates that the quantity of fossil fuels staying in the atmosphere is much less than previously thought. Evidence: (2007) concluded that unless energy policies changed substantially, the world would continue to depend on fossil fuels until 2025–2030.',
    'question_3': '"Global warming" and "climate change" mean different things and have both been used for decades. Evidence: Global warming and climate change are often used interchangeably.',
    'answer_1': 'SUPPORTS',
    'answer_2': 'NOT_ENOUGH_INFO',
    'answer_3': 'REFUTES',
}

# Default chat tokens, end prompts, and suffixes for each model
MODEL_CHAT_TOKENS = {
    'openai': '',
    'mistral': '<s>[INST] ',
    'llama-2': '<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n',
    'falcon': 'A helpful assistant.\nUser: ',
    'opt-finetune': '',
}

MODEL_END_PROMPTS = {
    'openai': ' What is the correct label?',
    'mistral': ' What is the correct label? [/INST]',
    'llama-2': ' What is the correct label? [/INST]',
    'falcon': ' What is the correct label?\nAssistant:',
    'opt-finetune': ' What is the correct label?',
}

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>',
    'opt-finetune': '</s>',
}

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def compute_openai_token_counts(bot: DialogueBot,
                          data: Iterable, 
                          input_column: str='concat_col',
                          max_samples: int=None,
                          start_prompt: str='Summarize the following: ',
                          end_prompt: str='\n Begin summary:') -> dict:
    """
    Compute the token count of an OpenAI model in a few-shot or zero-shot single-turn setting.
    """

    if not max_samples:
        max_samples = len(data)
    
    total_tokens = 0
    
    # Iterate over the test set
    for idx in range(max_samples):

        # Create the input string, adding the start and end prompts
        input = start_prompt + data[idx][input_column] + end_prompt
        true_input = bot.system_prompt + bot.history + [{'role': 'user', 'content': input}]
        num_tokens = num_tokens_from_messages(true_input, model=bot.model)
        total_tokens += num_tokens

    return total_tokens

def compute_openai_tokens_multiturn(bot: DialogueBot,
                          data: Iterable, 
                          input_column: str='concat_col',
                          max_samples: int=None,
                          chat: Iterable=None) -> dict:
    """
    Compute the token count of an OpenAI model in a multi-turn setting.
    """

    if not max_samples:
        max_samples = len(data)
    
    total_tokens = 0

    # Iterate over the test set
    for idx in range(max_samples):

        # Create the input string, adding the start and end prompts
        input = list(chat) + [{'role': 'user', 'content': data[idx][input_column]}]
        true_input = bot.system_prompt + bot.history + input
        num_tokens = num_tokens_from_messages(true_input, model=bot.model)
        total_tokens += num_tokens

    return total_tokens

# Main function
if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset requiring classification.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)', default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate', default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use in the results file', default='gpt-4-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='amandakonet/climate_fever_adopted')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default='')
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='train')
    parser.add_argument('--input_column', type=str, help='The name of the input column in the dataset', default='concat_col')
    parser.add_argument('--target_column', type=str, help='The name of the target column in the dataset', default='evidence_label')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=None)

    # Handle case where claim and evidence is separated
    parser.add_argument('--separate_inputs', type=bool, help='Specify if multiple input columns should be concatenated', default=True)
    parser.add_argument('--firstinput', type=str, help='Name of the first input column', default='claim')
    parser.add_argument('--firstinputflag', type=str, help='Name of the first input column', default='Claim: ')
    parser.add_argument('--secondinput', type=str, help='Name of the first input column', default='evidence')
    parser.add_argument('--secondinputflag', type=str, help='Name of the first input column', default='Evidence: ')

    # Prompt arguments
    parser.add_argument('--use_model_prompt_defaults', type=str, help='Whether to use the default prompts for a model', default='openai')
    parser.add_argument('--system_prompt', type=str, help='The system prompt for the model', default='You are a helpful assistant.')
    parser.add_argument('--start_prompt', type=str, help='The start prompt for the model', default=FACT_CHECK_INSTRUCTION)
    parser.add_argument('--end_prompt', type=str, help='The end prompt for the model', default=' Label?')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=974)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)
    parser.add_argument('--remove_stop_tokens', type=str, help='Stop tokens to remove from generated output separated by +', default='.')

    # Few-shot arguments
    parser.add_argument('--multiturn', type=bool, help='Whether to prompt in a multiturn setting', default=False)
    parser.add_argument('--shots', type=int, help='The number of shots to use for the model', default=0)
    parser.add_argument('--first_shot', type=str, help='The first shot to use for the model', default=FACT_CHECK_ONE_SHOT)
    parser.add_argument('--second_shot', type=str, help='The second shot to use for the model', default=FACT_CHECK_TWO_SHOT)
    parser.add_argument('--third_shot', type=str, help='The third shot to use for the model', default=FACT_CHECK_THREE_SHOT)
    parser.add_argument('--transition', type=str, help='The transition to use between shots', default=FACT_CHECK_TRANSITION)

    # PEFT arguments
    parser.add_argument('--peft_model', type=bool, help='Whether to use a PEFT model', default=False)
    parser.add_argument('--peft_dir', type=str, help='The path to the PEFT model config file', default='')
    parser.add_argument('--four_bit', type=bool, help='Whether to use a 4-bit PEFT model', default=True)
    parser.add_argument('--eight_bit', type=bool, help='Whether to use an 8-bit PEFT model', default=False)

    # Generation arguments
    parser.add_argument('--min_new_tokens', type=int, help='The minimum number of new tokens to generate', default=1)
    parser.add_argument('--max_new_tokens', type=int, help='The maximum number of new tokens to generate', default=10)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cuda:0')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--results_dir', type=str, help='The directory to save the results to', default='results')
    parser.add_argument('--intermediate_outputs_dir', type=str, help='The directory to save the intermediate outputs to', default='intermediate_outputs')
    parser.add_argument('--run_name', type=str, default='fact_checking_eval', help='The name of the project, for logging.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Update the start and end prompts if using the model defaults
    if args.use_model_prompt_defaults:

        args.start_prompt = MODEL_CHAT_TOKENS[args.use_model_prompt_defaults] + args.start_prompt
        args.end_prompt = MODEL_END_PROMPTS[args.use_model_prompt_defaults]
        args.remove_suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    
    if args.shots > 0 and args.multiturn:

        if args.shots == 1:
            chat = [
                {"role": "user", "content": args.start_prompt + MULTI_TURN_DICT['question_1']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_1']},
            ]
        elif args.shots == 2:
            chat = [
                {"role": "user", "content": args.start_prompt + MULTI_TURN_DICT['question_1']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_1']},
                {"role": "user", "content": MULTI_TURN_DICT['question_2']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_2']},    
            ]
        elif args.shots == 3:
            chat = [
                {"role": "user", "content": args.start_prompt + MULTI_TURN_DICT['question_1']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_1']},
                {"role": "user", "content": MULTI_TURN_DICT['question_2']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_2']},    
                {"role": "user", "content": MULTI_TURN_DICT['question_3']},
                {"role": "assistant", "content": MULTI_TURN_DICT['answer_3']},                    
            ]
        else:
            raise ValueError('Invalid number of shots: ', args.shots)
 
    # Add shots to the start prompt if specified
    elif args.shots > 0:

        if args.shots == 1:
            args.start_prompt = args.start_prompt + args.first_shot + args.transition
        elif args.shots == 2:
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.transition
        elif args.shots == 3:
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.third_shot + args.transition
        else:
            raise ValueError('Invalid number of shots: ', args.shots)

    # Create list of stop tokens to remove
    if args.remove_stop_tokens:
        args.remove_stop_tokens = args.remove_stop_tokens.split('+')
    
    # Load the test split of the dataset
    print('Loading dataset: ', args.dataset)

    if args.dataset_revision:
        data = load_dataset(args.dataset, args.dataset_revision, split=args.split)

    else:
        data = load_dataset(args.dataset, split=args.split)

    # If input uses multiple columns, concatenate into a new column
    if args.separate_inputs:

        concat_col = [f'{args.firstinputflag}{data[args.firstinput][i]} {args.secondinputflag}{data[args.secondinput][i]}' for i in range(len(data))]
        data = data.add_column('concat_col', concat_col)
        args.input_column = 'concat_col'

    args.intermediate_outputs_dir = f'{args.intermediate_outputs_dir}_{args.shots}-shot'


    bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)

    if not args.multiturn:
        tokens = compute_openai_token_counts(bot,
                                        data, 
                                        args.input_column, 
                                        args.max_samples, 
                                        args.start_prompt,
                                        args.end_prompt)

    else:
        tokens = compute_openai_tokens_multiturn(bot,
                                        data, 
                                        args.input_column, 
                                        args.max_samples, 
                                        chat)

    print('Total tokens: ', tokens)
