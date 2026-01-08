import os
import json
import csv
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------
# config
# ---------------------------
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_DIR = "./phi3-genetics-lora_v0"  

OUT_JSONL = "phi3_lora_20answers_T60_20251215.jsonl"
OUT_CSV = "phi3_lora_20answers_T60_20251215.csv"

SYSTEM_PROMPT = (
    "You are an expert assistant in human genetics and genomics. "
    "Answer in a clinically useful, structured way. "
    "Be accurate and do not invent facts. If evidence is insufficient, say so."
)

QUESTIONS = [
    "Under what circumstances can a high allele frequency in gnomAD be used as strong evidence against pathogenicity for a rare disease?",
    "How do you determine whether a premature stop-gain variant qualifies for PVS1 (very strong) versus only PVS1-moderate under ACMG guidelines?",
    "What factors determine whether a de novo variant can be used as strong evidence (PS2) rather than moderate (PM6)?",
    "How should conflicting in-silico predictions (e.g., CADD high, REVEL low) be handled during missense variant interpretation?",
    "What population-genetics parameters must be considered before applying BA1 or BS1 to rule out pathogenicity?",
    "How does gene-specific constraint (pLI/LOEUF) inform the interpretation of missense and LOF variants?",
    "When can segregation analysis be used as strong evidence for pathogenicity (PP1-Strong), and what limitations must be met?",
    "How do you differentiate between a benign and pathogenic splice-region variant when RNA studies are not available?",
    "What criteria must be satisfied for a variant to be classified as a founder mutation in a specific population?",
    "How does variable penetrance affect the clinical interpretation of a pathogenic variant in an autosomal-dominant disorder?",
    "What distinguishes a pathogenic gain-of-function variant from a haploinsufficiency-driven loss-of-function variant?",
    "When interpreting CNVs, how do you determine whether gene dosage (triplosensitivity or haploinsufficiency) is relevant?",
    "How should variants located in low-complexity or segmentally duplicated regions be validated before interpretation?",
    "In recessive disorders, how do you evaluate pathogenicity when one allele is a known pathogenic variant and the second is a VUS?",
    "What evidence supports reclassification of a VUS when multiple unrelated patients present with the same variant and overlapping phenotypes?",
    "How does transcript selection (MANE Select vs tissue-specific isoforms) impact the interpretation of coding variants?",
    "What are the key criteria for interpreting mosaic variants in a dominant disorder with high de-novo rates?",
    "How should you interpret a missense variant occurring in a mutational hotspot or a critical domain of a protein?",
    "Under what conditions can in-vitro functional data be used as strong evidence (PS3), and what are the quality requirements?",
    "How do you determine whether a variant identified by tumor sequencing is germline, somatic, or ambiguous for clinical reporting?"
]

# Generation params
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False          
TEMPERATURE = 0.0          
TOP_P = 1.0
REPETITION_PENALTY = 1.0


def safe_mkdir_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_model_and_tokenizer():
    # Avoid tokenizers fork warnings / deadlocks in some cluster envs
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    adapter_abs = os.path.abspath(LORA_ADAPTER_DIR)
    if not os.path.isdir(adapter_abs):
        raise FileNotFoundError(f"LoRA adapter folder not found: {adapter_abs}")
    if not os.path.exists(os.path.join(adapter_abs, "adapter_config.json")):
        raise FileNotFoundError(
            f"adapter_config.json not found in: {adapter_abs}\n"
            f"Contents: {os.listdir(adapter_abs)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter_abs, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL warned about left padding; right padding is safer in fp16/bf16 training/generation
    tokenizer.padding_side = "right"

    # Base model
    dtype = torch.float16
    if torch.cuda.is_available():
        # If BF16 is supported, we can swap to bfloat16; fp16 is fine for generation.
        dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=False, 
        device_map="auto",
        dtype=torch.float16
        #trust_remote_code=True,
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_abs)
    model.eval()

    return model, tokenizer, adapter_abs


def generate_answer(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            repetition_penalty=REPETITION_PENALTY,
        )

    # Only decode newly generated tokens
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer


def main():
    model, tokenizer, adapter_abs = load_model_and_tokenizer()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at = datetime.now(timezone.utc).isoformat()

    results = []
    for i, q in enumerate(QUESTIONS, start=1):
        ans = generate_answer(model, tokenizer, q)
        results.append({
            "run_id": run_id,
            "created_at_utc": created_at,
            "question_id": i,
            "question": q,
            "answer": ans,
            "base_model": BASE_MODEL_ID,
            "lora_adapter_dir": adapter_abs,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
        })
        print(f"[{i:02d}/{len(QUESTIONS)}] done")

    # Save JSONL
    safe_mkdir_parent(OUT_JSONL)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save CSV
    safe_mkdir_parent(OUT_CSV)
    fieldnames = list(results[0].keys()) if results else []
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print("\n Saved:")
    print(" -", os.path.abspath(OUT_JSONL))
    print(" -", os.path.abspath(OUT_CSV))


if __name__ == "__main__":
    main()
