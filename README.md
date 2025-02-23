# qlora-finetune-colab

A lightweight implementation for QLoRA fine-tuning of large language models in Google Colab. Tested with Mistral-7B-Instruct-v0.3. Originally wrote it for Blocktail methodology benchmarking.

## Quick Start

1. Open `finetune_mistral.ipynb` in Colab
2. Add your HF token to Colab secrets as `HF_TOKEN`
3. Upload your JSONL files to `/content/data/` directory in Colab
4. Run all cells sequentially (memory cleanup is important)

## Training Approach

- Uses QLoRA (Quantized Low-Rank Adaptation) for memory-efficient training
- Target modules: q_proj, k_proj, v_proj, o_proj (attention layers)
- Rank (r): 32 (LoRA dimension)
- Alpha: 64 (LoRA scaling)
- 4-bit NF4 quantization with nested quantization
- Gradient checkpointing enabled
- Cosine learning rate scheduling with warmup

## Training Parameters

- Learning rate: 1e-4
- Epochs: 5
- Batch size: 4 per device
- Gradient accumulation steps: 4
- Max sequence length: 4096 tokens
- Warmup ratio: 0.1

## Paths in Colab

- Training data: `/content/data/*.jsonl`
- Model cache: `/content/model_cache`
- Training outputs: `/content/model_outputs`
- Final model: `/content/model_outputs/{run_name}/final_model`
- Merged model (optional): `/content/merged_mistral_7b`

## Model Output

This training produces LoRA adapter weights, not a full model. To use the fine-tuned model, you'll need:
1. The original base model (e.g., Mistral-7B-Instruct-v0.3)
2. The trained LoRA adapters (saved to HF Hub and locally)

You can either:
- Use the LoRA adapters for inference (requires PEFT)
- Merge the adapters with the base model for a standalone model
- Load both the base model and adapters at runtime for memory efficiency

## Features

- QLoRA implementation with 4-bit quantization
- Automatic HuggingFace Hub integration (private repo by default)
- Progress tracking with TensorBoard
- Local backup saves after training
- BFloat16 precision for A100 GPU
- Memory management utilities included

> we enabled `bf16=True` in our training arguments for GPUs with native bfloat16  (*e.g., A100*). this helps stabilize training by allowing a bit wider numeric range. if you prefer to keep everything strictly in FP16—or if your GPU doesn’t support BF16—simply set `bf16=False`.

## Dependencies
```
transformers>=4.41.0,<5.0.0
accelerate==0.26.0
peft
bitsandbytes
datasets
trl
```

## Data Format

JSONL format:
```json
{"prompt": "What is X?", "completion": "X is Y"}
```

Formatted as:
```
### Question: {prompt}

### Answer: {completion}

### End
```

## Requirements

- Google Colab (A100 GPU recommended)
- HuggingFace account and token
- Training data in JSONL format