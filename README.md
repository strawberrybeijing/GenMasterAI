# Phi-3 QLoRA Fine-Tuning to Genetic based

This repository contains scripts to benchmark the **raw Phi-3 model**, fine-tune it using **QLoRA**, and evaluate the model **after LoRA adaptation** on different question sets, including obesity-related variants.

## Environment Setup

All dependencies are managed using Conda:

- conda env create -f transformers_env.yml
- conda activate transformers_env

## Running on GPU (UNIL Curnagl Cluster)

1. Interactive GPU session:
- srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem 8G --time=3:00:00 --pty bash 
- conda activate transformers_env
- python xxxx.py

2.Batch job submission:
- A Slurm submission script is provided: **launch_code_gpu.slurm**
- sbatch launch_code_gpu.slurm


## Files Overview

### 1. `raw_model_benchmark_phi3_20questions_20260108_JZ.py`
Benchmark script for the **raw (base) Phi-3 model**.

- Evaluates the original model before any fine-tuning  
- Uses a set of **20 general questions**  
- Serves as the baseline for comparison with the LoRA-fine-tuned model

---

### 2. `raw_model_benchmark_phi3_5variants_obesity_20260108_JZ.py`
Benchmark script for the **raw Phi-3 model** on a domain-specific task.

- Tests the base model on **5 obesity-related question variants**  
- Measures baseline performance and response consistency  
- Used to compare against post-LoRA results

---

### 3. `qlora_phi3_20260108_JZ.py`
QLoRA fine-tuning script for the Phi-3 model.

- Implements **QLoRA (Quantized Low-Rank Adaptation)**  
- Fine-tunes the base Phi-3 model using the provided dataset  
- Produces a LoRA adapter that can be loaded for inference

---

### 4. `after_lora_phi3_20questions_20260108_JZ.py`
Evaluation script for the **LoRA-adapted Phi-3 model**.

- Loads the trained **LoRA adapter**  
- Evaluates the model on the same **20-question set** used for the raw benchmark  
- Enables direct before/after LoRA comparison

---

### 5. `after_lora_phi3_5variantss_obesity_test_20260108_JZ.py`
Post-LoRA evaluation on obesity-related prompts.

- Loads the **LoRA adapter** trained with QLoRA  
- Tests the adapted model on **5 obesity-related question variants**  
- Assesses domain adaptation and generalization after fine-tuning

---

### 6. `60ds.jsonl`
Training dataset used for LoRA fine-tuning.

- JSONL-format dataset  
- Used as input for the QLoRA training script  
- Contains the examples used to adapt the Phi-3 model

### 7. `transformers_env.yml`
Conda env used during initial analysis

### 8. `launch_code_gpu.slurm`
launch job on GPU on UNIL Curnagl cluster

---

## Workflow Summary

1. Run raw model benchmark scripts to establish baseline performance  
2. Fine-tune the Phi-3 model using QLoRA  
3. Load the LoRA adapter and run post-LoRA evaluation scripts  
4. Compare raw vs. LoRA-adapted model outputs
