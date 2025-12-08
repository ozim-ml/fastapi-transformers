from transformers import pipeline
import torch
import os

device_index = int(os.environ.get("DEVICE_INDEX", "0"))  

clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device_index,
    batch_size=16,
    model_kwargs={"dtype": torch.float16} if device_index >= 0 else {},
)