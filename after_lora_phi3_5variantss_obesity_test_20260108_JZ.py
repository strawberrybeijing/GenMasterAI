import os
import json
import csv
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------
# User config
# ---------------------------
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_DIR = "./phi3-genetics-lora_v0"  # folder of adapter_config.json

OUT_JSONL = "phi3_lora_5variants_Obesity_20260108.jsonl"
OUT_CSV = "phi3_lora_5variants_Obesity_20260108.csv"

SYSTEM_PROMPT = (
    "You are an expert assistant in human genetics and genomics. "
    "Answer in a clinically useful, structured way. "
    "Be accurate and do not invent facts. If evidence is insufficient, say so."
    "You must Comment every variant, check for frequency, Clinvar class, Mode of Inheritence and CADD."
    "You Must report the Allelic balance and the Frequency of each variant, if no frequency mentioned in the input table, it is 0"
)

QUESTIONS = [
"""Here is a table contains 5 variants. Wchich variant cause Obesity? 
Variant 1: Variant=chr19:40730680_1 insG, Chromosome=chr19, Position=40730681, RS_ID=NA, Ref_seq=NA, Var_seq=G, Type=Insertion, HGVS=NM_024877.4:c.305dup p.(Val103CysfsTer25), Zygosity=Heterozygous, Gene=CCNP, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinVar_class=NA, Allelic_balance=0.5417, Frequency=NA, CADD_score=NA
Variant 2: Variant=chr16:56904031 C>T, Chromosome=chr16, Position=56904031, RS_ID=rs28936388, Ref_seq=C, Var_seq=T, Type=SNV, HGVS=NM_001126108.2:c.625C>T p.(Arg209Trp), Zygosity=Heterozygous, Gene=SLC12A3, OMIM_phenotype=Gitelman syndrome, OMIM_inheritance=Autosomal recessive, Inheritance=AR, ClinVar_class=Pathogenic, Allelic_balance=0.5393, Frequency=1.19E-05, CADD_score=26
Variant 3: Variant=chr12:117962872 G>A, Chromosome=chr12, Position=117962872, RS_ID=rs56178407, Ref_seq=G, Var_seq=A, Type=SNV, HGVS=NM_173598.6:c.2004C>T p.(Ile668=), Zygosity=Heterozygous, Gene=KSR2, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinVar_class=Benign, Allelic_balance=0.3947, Frequency=0.00689813, CADD_score=5.871
Variant 4: Variant=chr3:49690627 G>A, Chromosome=chr3, Position=49690627, RS_ID=rs35762866, Ref_seq=G, Var_seq=A, Type=SNV, HGVS=NM_003458.4:c.3638G>A p.(Gly1213Asp), Zygosity=Heterozygous, Gene=BSN, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinVar_class=Benign, Allelic_balance=0.5556, Frequency=0.0813352, CADD_score=12.92
Variant 5: Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, RS_ID=NA, Ref_seq=T, Var_seq=C, Type=SNV, HGVS=NM_005068.3:c.744-2A>G p.?, Zygosity=Heterozygous, Gene=SIM1, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=AD, ClinVar_class=NA, Allelic_balance=0.383, Frequency=NA, CADD_score=34
"""
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

    # Tokenizer: you saved one in adapter dir; this is fine.
    tokenizer = AutoTokenizer.from_pretrained(adapter_abs, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL warned about left padding; right padding is safer in fp16/bf16 training/generation
    tokenizer.padding_side = "right"

    # Base model
    dtype = torch.float16
    if torch.cuda.is_available():
        # If BF16 is supported, you can swap to bfloat16; fp16 is fine for generation.
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

    print("\n✅ Saved:")
    print(" -", os.path.abspath(OUT_JSONL))
    print(" -", os.path.abspath(OUT_CSV))


if __name__ == "__main__":
    main()
