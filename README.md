# Email Classification API using FastAPI and Transformers

This project provides a FastAPI-based REST API for classifying email content using a Large Language Model via HuggingFace Transformers.
Powered by the BART-Large-MNLI language model, it performs zero-shot classification across multiple tasks.

The API provides sentence chunking and returns both task-level and chunk-level predictions.
Deployment uses Docker, CUDA, and Poetry to ensure a fully reproducible environment.