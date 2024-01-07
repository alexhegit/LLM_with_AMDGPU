import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

pmodel="../LLM_Files/Baichuan2-13B-Chat/"

tokenizer = AutoTokenizer.from_pretrained(pmodel, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pmodel, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(pmodel)
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)
