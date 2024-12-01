import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
token = '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'
huggingface_hub.login(token=token)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Dowdloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
print("Dowdloading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16
)

save_directory = "gemma_model"
print("Saving tokenizer...")
tokenizer.save_pretrained(save_directory)
print("Saving model...")
model.save_pretrained(save_directory)

print("Done")
