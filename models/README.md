# LLM Models Directory

This directory is used to store local LLM models for the FileSorter application. The application supports various local LLM models including Mistral, Llama, and others.

## Model Installation

1. Download your preferred GGUF model files from Hugging Face:
   - [Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF)
   - [Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)

2. Place the downloaded model files in this directory with the following names:
   - Mistral-7B: `mistral-7b.gguf`
   - Llama-2: `llama-2.gguf`
   - Grok-1: `grok-1.gguf` (when available)

## Supported Models

The application currently supports the following models:
- Mistral-7B
- Llama-2
- Grok-1 (pending availability)

## Model Configuration

The models are loaded with CPU-only configuration for maximum compatibility. For better performance on systems with capable GPUs, modify the `gpu_layers` parameter in the `LLMManager` class.
