import os
import numpy as np
import torch
import openai
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes  # Works with CUDA

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class OpenAIModel():
    # Ref: https://platform.openai.com/docs/api-reference/making-requests
    def __init__(self, api_key, model_name='gpt-3.5-turbo', system_prompt=None):
        # Model: gpt-4, gpt-4 turbo, gpt-3.5-turbo
        self.model_name = model_name
        self.system_prompt = 'You are a helpful assistant.' if system_prompt is None else system_prompt
        self.client = openai.OpenAI(api_key=api_key)

    def __call__(self, prompt, **kwds):
        # Possible parameters: https://platform.openai.com/docs/api-reference/chat/create?lang=python
        messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content':prompt}]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages)

        return response.choices[0].message.content
    

class AnthropicModel():
    # Ref: https://github.com/anthropics/anthropic-sdk-python
    def __init__(self, api_key, model_name='claude-2'):
        self.model_name = model_name
        self.anthropic = Anthropic(api_key=api_key)

    def __call__(self, prompt, **kwds):
        response = anthropic.completions.create(model=self.model_name, prompt=f'{HUMAN_PROMPT} {prompt}{AI_PROMPT}')
        return response.completion


# class GoogleModel():
#     # Waiting to get access to oficial API
#     def __init__(self, api_key, model_name):
#         self.model_name = model_name
#         self.api_key = api_key

#     def __call__(self, prompt, **kwds):
#         return response


class HFModel():
    def __init__(self, api_key, model_name, task_name=None, tokenizer_name=None):
        self.model_name = model_name
        if 'llama-2-7b' in model_name.lower():
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config,  token=api_key)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=api_key)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=api_key)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=api_key)

    def __call__(self, prompt, **kwds):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)  #Use return_length to return input lenght in tokens
        output = self.model.generate(**model_inputs, output_scores=True, return_dict_in_generate=True);  #True to return prediction scores, format return_dict_in_generate
        response = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return response

def get_model(api_key, model_name, system_prompt=None, task_name=None, tokenizer=None):
    if model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
        model = OpenAIModel(api_key, model_name=model_name, system_prompt=system_prompt)
    elif model_name in ['claude-2', 'claude-instant-1']:
        model = AnthropicModel(api_key, model_name=model_name)
    # elif model_name == 'bard':
    #     model = GoogleModel(api_key)
    else:
        model = HFModel(api_key, model_name, task_name)
        # raise NotImplementedError('Model not supported.')
    return model
      
    
class SafetyCheck():
    def __init__(self, method='keywords'):
        self.method = method

        if method == 'keywords':
            self.key_words = self.load_keywords()
        elif method == 'toxigen-hatebert':
            # From: https://github.com/microsoft/TOXIGEN
            self.model = pipeline('text-classification', model='tomh/toxigen_hatebert', tokenizer='bert-base-cased')
        elif method == 'toxigen-roberta':
            # From: https://github.com/microsoft/TOXIGEN
            self.model = pipeline('text-classification', model='tomh/toxigen_roberta')
        else:
            raise NotImplementedError('Method not supported.')

    def __call__(self, text, **kwds):
        if self.method == 'keywords':
            return self.check_keywords(text)
        elif self.method in ['toxigen-hatebert', 'toxigen-roberta']:
            return self.check_toxigen(text)
        
    def load_keywords(self):
        f = open('data/safety_keywords.txt', 'r')
        keyword_list = f.readlines()
        return set(keyword_list)

    def check_keywords(self, text):
        # From: https://github.com/llm-attacks/llm-attacks/blob/main/api_experiments/evaluate_api_models.py
        score = np.mean([x not in text for x in self.key_words])
        result = 'Passed' if score == 1 else 'Failed' 
        return result, score
    
    def check_toxigen(self, text):
        self.model.to(device) # Use GPU
        output = self.model(text)
        score = output[0]['score'] # Probability / confidence of the model
        result = output[0]['label'] 
        return result, score
    