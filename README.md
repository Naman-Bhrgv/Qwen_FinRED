# QLoRA Fine-Tuning of Qwen2.5-3B on FinRED (Financial Relation Extraction)

This notebook fine-tunes Qwen/Qwen2.5-3B with QLoRA (4-bit NF4 + LoRA) on the
FinGPT/fingpt-finred-re dataset to perform financial relation extraction.

## Requirements

Single CUDA GPU (tested configuration targets Kaggle T4/P100; any single-GPU CUDA
setup should work) <br>
Python packages (installed by the notebook itself): <br>
transformers>=4.46.0, peft>=0.13.0, bitsandbytes>=0.44.0, accelerate>=1.0.0, <br>
datasets>=3.0.0, trl>=0.11.0, evaluate, scikit-learn, sentencepiece <br>
torch is expected to already be installed in the environment <br>

## Output <br>

Adapters and training artifacts are written under: <br>

<code>
/kaggle/working/qlora_finred/r<rank>/
├── (Trainer checkpoints, if any)
└── adapter/            # saved LoRA adapter (model.save_pretrained)
</code>
<br>
The final cell produces a results_df pandas DataFrame with one row per run, containing:
rank, alpha, trainable_params, trainable_pct, adapter_mem_tensor_mb,
adapter_mem_disk_mb, peak_gpu_mem_mb, train_time_min, precision, recall, f1,
n_eval.
