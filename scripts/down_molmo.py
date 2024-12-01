import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from typing import List, Optional, Union, Mapping

custom_model_path = "/home/ltnghia02/models/molmo_model"

processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-72B-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
    cache_dir=custom_model_path,
    resume_download=True
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-72B-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
    max_position_embeddings=8192,
    # max_memory = {0: "80GB", 1: "80GB", 2: "80GB", 3: "80GB",},
    cache_dir=custom_model_path,
    resume_download=True
)
