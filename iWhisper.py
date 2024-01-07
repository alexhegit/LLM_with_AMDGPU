#Refer to https://huggingface.co/openai/whisper-large-v3

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: "+device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#model_id = "openai/whisper-large-v3"
model_id = "../LLM_Files/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

#Use BetterTransformers with pip optimum package
#model = model.to_bettertransformer()

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]
#result = pipe(sample)

result = pipe("../Jay_m.mp3")
print(result["text"])

