#!/usr/bin/env python3

import os
import evaluate
import numpy as np
import json
import argparse
import torch
import wandb
from collections import Counter
import argparse
import string
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv
# from openai_chat_api import DialogueBot
# from utils import (
#     generate_completions,
#     load_hf_lm_and_tokenizer,
#     query_openai_chat_model,
#     dynamic_import_function,
# )
def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def extract_answer(predict_answer):
    predict_answer = predict_answer.replace("</s>","")
    predict_answer = predict_answer.replace("<pad>","")
    predict_answer = predict_answer.replace("/n","")
    predict_answer = predict_answer.replace("\\","")
    if "unanswerable" in predict_answer.lower():
        predict_answer = "unanswerable"
    elif predict_answer.lower().startswith("yes") or predict_answer.lower().startswith(" yes"):
        predict_answer = "yes"
    elif predict_answer.lower().startswith("no") or predict_answer.lower().startswith(" no"):
        predict_answer = "no"
    # print(predict_answer)
    return predict_answer


def get_answers_and_evidence(data):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    references.append({"answer": answer, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def evaluate_f1(gold, predicted):
    hashas = 0.0
    nono = 0.0
    hasno = 0.0
    nohas = 0.0
    max_answer_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    unanswerable_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }

    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            # max_answer_f1s.append(0.0)
            continue
        answer = extract_answer(predicted[question_id])

        answer_f1s_and_types = [
            (token_f1_score(answer, reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]

        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]


        if "unanswerable" in answer.lower():
            unanswerable_by_type[answer_type].append(1.0)
        else:
            unanswerable_by_type[answer_type].append(0.0)



        # f1.write(json.dumps({
        #     'question_id': question_id,
        #     'predicted_answer': answer,
        #     'gt': [each['answer'] for each in gold[question_id]],
        #     'f1':answer_f1s_and_types
        # }) + "\n")
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)


    mean = lambda x: sum(x) / len(x) if x else 0.0
    all = []
    for i in list(unanswerable_by_type.values()):
        all += i

    has_all = []
    for key, value in unanswerable_by_type.items():
        if key != "none":
            has_all += value

    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Answerability_all": mean(all),
        "Answerability_has": mean(has_all),
        "noans F1 by type": {key: mean(value) for key, value in unanswerable_by_type.items()},
    }

def compute_summarization_metrics(predictions: Iterable, 
                            references: Iterable,
                            predictions_dict:dict,
                            rouge: bool=True,
                            bleu: bool=True,
                            f1: bool=True,
                            bertscore: bool=True) -> dict:
    """
    Compute ROUGE, BLEU, and BERTscore metrics for a set of predictions and references.
    """

    metric_results = {}
    predictions = [extract_answer(i) for i in predictions]

    if rouge:
        rouge = evaluate.load('rouge')

        # Compute ROUGE metrics at the summary level, using the 'rouge1', 'rouge2', and 'rougeL' metrics, aggregating the results
        rouge_results = rouge.compute(predictions=predictions, 
                                    references=references, 
                                    use_aggregator=True)

        # Store the results in the metric_results dictionary
        metric_results['rouge'] = rouge_results
    
    else:
        metric_results['rouge'] = None

    if bleu:
        bleu = evaluate.load('bleu')

        # Compute BLEU metrics at the summary level
        bleu_results = bleu.compute(predictions=predictions, 
                                    references=references)
        
        # Store the results in the metric_results dictionary
        metric_results['bleu'] = bleu_results
    
    else:
        metric_results['bleu'] = None

    if bertscore:
        bertscore = evaluate.load('bertscore')

        # Compute BERTscore metric, using distilbert-base-uncased as the reference model, and averaging the results
        bertscore_results = bertscore.compute(predictions=predictions, 
                                                    references=references, 
                                                    lang='en', 
                                                    model_type="distilbert-base-uncased")
        
        # Store the results in the metric_results dictionary
        metric_results['bertscore'] = {k: np.mean(v) for k, v in bertscore_results.items() if k in ['precision', 'recall', 'f1']}
    
    else:
        metric_results['bertscore'] = None


    if f1:
        # base_url = "/Users/wenbingbing/PycharmProjects/lab-scale-ai/tasks/"
        base_url = "/home/bingbw/fine_tuning/data/"
        gold_data = json.load(open(base_url+"qasper-test-v0.3.json"))
        gold_answers_and_evidence = get_answers_and_evidence(gold_data)
        evaluation_output = evaluate_f1(gold_answers_and_evidence, predictions_dict)
        metric_results['f1'] = evaluation_output
    else:
        metric_results['f1'] = None




    return metric_results

def generate_from_prompt(model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         input_data: str,
                         max_tokens: int=2048,
                         min_new_tokens: int=1,
                         max_new_tokens: int=50) -> str:
    """
    Generate and decode output from a Transformers model using a prompt.
    """

    # Check whether input will not include the end prompt due to context window length, and manually truncate if necessary
    tokenized = tokenizer.encode(input_data)

    # If the input is too long, truncate it to the maximum length minus the length of the end prompt
    #if len(tokenized) > max_tokens:
    #  input = tokenizer.decode(tokenized[:max_tokens-len(tokenizer.encode(end_prompt))-1], skip_special_tokens=True) + end_prompt

    # Calculate the position of the start of the output string
    start_decode = len(tokenizer.encode(input_data, truncation=True, max_length=max_tokens))

    # Encode the input string
    input_ids = tokenizer(input_data, return_tensors='pt', truncation=True, max_length=max_tokens).to(model.device)

    # Generate text from prompt
    with torch.no_grad():
      output = model.generate(**input_ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
    
    # Decode the output string, removing the special tokens and any suffixes
    decoded = tokenizer.decode(output[0][start_decode:])

    return decoded
                             
def evaluate_hf_model( model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      args,
                      examples,
                      shot: str='zero',
                      max_samples: int=None,
                      system_message: str='###',
                      transaction: str='###',
                      max_tokens: int=2048,
                      min_new_tokens: int=1,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      rouge: bool=True,
                      bleu: bool=True,
                      bertscore: bool=True,
                      f1:bool=True,
                      ) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []
    model_outputs_dict = {}

                          
    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating Hugging Face model'):
  
        # if args.no_context:
        #     instruction = f"""\n\nCreate an Answer to the Question. Pay attention to answer only \"yes\" or \"no\" for boolean questions.  Answer \"Unanswerable\" when you are not sure about the answer."""
        #     test_question = f"""\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""
        # else:
        #     instruction = f"""\n\nCreate an Answer to the Question using following documents. Pay attention to answer only \"yes\" or \"no\" for boolean questions.  Answer \"Unanswerable\" when you are not sure about the answer."""
        #     test_question = f"""\n\n## Documents:\n{data['context'][idx]}\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""
        #instruct-following b
        if args.no_context:
            instruction = f"""\n\nCreate an Answer to the Question. Answer \"Unanswerable\" when you are not sure about the answer."""
            test_question = f"""\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""
        else:
            instruction = f"""\n\nCreate an Answer to the Question using following documents. Answer \"Unanswerable\" when you are not sure about the answer."""
            test_question = f"""\n\n## Documents:\n{data['context'][idx]}\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""

        # prompt = instruction+ examples + test_question
        # prompt = examples + instruction+test_question
        chat = [{"role": "user", "content": system_message+
                                        examples+
                                        transaction+
                                        instruction+
                                        test_question}]

        input_data = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if 'falcon' in tokenizer.name_or_path:
            input_data = input_data.replace('user', 'User:')
            input_data = input_data.replace('assistant', 'Assistant:')
            input_data = input_data.replace('Assistant:!', 'assistant!')
            input_data = input_data.replace('<|im_start|>', '')
            input_data = input_data.replace('<|im_end|>', '')
            
        ## decoding
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=input_data, 
                                       max_tokens=max_tokens,
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        model_outputs.append(decoded)
        model_outputs_dict[data['question_id'][idx]] = decoded

    print(f'{shot} Results:')
    if args.pretrain == 'True':
        np.save(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_pretrained_model_{shot}shot_outputs_{args.no_context}.npy",
                model_outputs)
    else:
        np.save(f"/home/bingbw/fine_tuning/results/{args.use_model_prompt_defaults}_finetuned_model_{shot}shot_outputs_{args.no_context}.npy", model_outputs)

    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries    
    metrics = compute_summarization_metrics(model_outputs,
                                            data['answer'][:len(model_outputs)],
                                            model_outputs_dict,
                                            # rouge=rouge,
                                            # bleu=bleu,
                                            # bertscore=bertscore,
                                            f1=f1)


    return model_outputs, metrics

# def evaluate_openai_model(openai_engine,
#                           data,
#                           args,
#                           examples,
#                           max_samples,
#                           shot,
#                           eval_batch_size=1,
#                           rouge: bool = True,
#                           bleu: bool = True,
#                           bertscore: bool = True,
#                           f1: bool = True,
#                           ) -> dict:
#     """
#     Evaluate an OpenAI model on a dataset using three classification metrics.
#     """
#     prompts = []
#
#     for idx in tqdm(range(max_samples), desc='Evaluating open ai model'):
#         # Generate and decode the output string, removing the special tokens and any suffixes
#         if args.no_context:
#             instruction = f"""\n\nCreate an Answer to the Question. Pay attention to answer only \"yes\" or \"no\" for boolean questions.  Answer \"Unanswerable\" when you are not sure about the answer."""
#             test_question = f"""\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""
#         else:
#             instruction = f"""\n\nCreate an Answer to the Question using following documents. Pay attention to answer only \"yes\" or \"no\" for boolean questions.  Answer \"Unanswerable\" when you are not sure about the answer."""
#             test_question = f"""\n\n## Documents:\n{data['context'][idx]}\n\n## Question:\n{data['question'][idx]}\n\n## Answer:"""
#
#         # prompt = instruction+ examples + test_question
#         prompt = examples + instruction+test_question
#         print(prompt)
#         prompts.append(prompt)
#     model_outputs_dict = {}
#
#     instances = [{"id": example["question_id"], "prompt": prompt} for example, prompt in zip(data, prompts)]
#     results = query_openai_chat_model(
#         engine=openai_engine,
#         instances=instances,
#         output_path=os.path.join("./results", f"{shot}shot_{openai_engine}_qasper_results_{args.no_context}.jsonl"),
#         batch_size=eval_batch_size,
#     )
#     model_outputs = [result["output"].strip().split("\n")[0].strip() for result in results]
#     for example, output in zip(data, model_outputs):
#         model_outputs_dict[example["question_id"]] = output
#     # Compute the classification metrics, comparing the model's responses to the target labels
#     metrics = compute_summarization_metrics(model_outputs,
#                                             data['answer'][:len(model_outputs)],
#                                             model_outputs_dict,
#                                             rouge=rouge,
#                                             bleu=bleu,
#                                             bertscore=bertscore,
#                                             f1=f1)
#     return model_outputs, metrics
