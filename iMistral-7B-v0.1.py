import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

pmodel="../LLM_Files/Mistral-7B-v0.1/"

model = AutoModelForCausalLM.from_pretrained(pmodel, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(pmodel, trust_remote_code=True)

#model = AutoModelForCausalLM.from_pretrained(pmodel, torch_dtype="auto", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(pmodel, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(pmodel, trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
