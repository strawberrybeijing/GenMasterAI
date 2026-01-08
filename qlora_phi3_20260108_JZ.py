import os
import argparse
import inspect
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def parse_args():
    p = argparse.ArgumentParser()

    # Model / data
    p.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--train_file", type=str, default="60ds.jsonl")
    p.add_argument("--output_dir", type=str, default="phi3-genetics-lora_v0")

    # QLoRA default
    p.add_argument("--no_4bit", action="store_true",
                   help="Disable 4-bit QLoRA; use standard LoRA (needs more VRAM).")

    # LoRA hyperparams (conservative defaults for small datasets)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.10)

    # Training
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=5.0)  # upper bound; early stopping will stop earlier
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Split / early stopping
    p.add_argument("--eval_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=2)

    # Logging / eval / save (important for tiny datasets!)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=5)
    p.add_argument("--save_total_limit", type=int, default=2)

    # Efficiency
    p.add_argument("--packing", action="store_true", default=True)

    return p.parse_args()


def choose_dtype():
    """dtype for GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    try:
        major, _ = torch.cuda.get_device_capability(0)
        # Ampere+ usually supports bf16 well
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return torch.float16


def build_training_args(args, use_4bit, fp16, bf16):
    """
    Transformers changed argument name from evaluation_strategy -> eval_strategy here Jing
    """
    kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_steps=args.logging_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,

        fp16=fp16,
        bf16=bf16,

        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Compatibility for eval
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎯 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Quantization (QLoRA 4-bit)
    # ----------------------------
    use_4bit = not args.no_4bit
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    dtype = choose_dtype()
    fp16 = (dtype == torch.float16)
    bf16 = (dtype == torch.bfloat16)

    # ----------------------------
    # Model loading
    # ----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ----------------------------
    # LoRA config
    # ----------------------------
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----------------------------
    # Load dataset (jsonl strcuture: instruction/input/output)
    # ----------------------------
    ds = load_dataset("json", data_files={"data": args.train_file})["data"]

    split = ds.train_test_split(test_size=args.eval_split, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    system_prompt = (
        "You are a helpful clinical genetics and bioinformatics assistant. "
        "Be accurate, structured, and do not invent facts. "
        "If evidence is insufficient, say so."
    )

    RESPONSE_TEMPLATE = "### Assistant:\n"

    def to_text(example):
        inst = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        user = inst if inp == "" else f"{inst}\n\n{inp}"
        text = (
            f"### System:\n{system_prompt}\n\n"
            f"### User:\n{user}\n\n"
            f"{RESPONSE_TEMPLATE}"
            f"{out}\n"
        )
        return {"text": text}

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

    # Mask loss so we only learn on assistant answers
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ----------------------------
    # TrainingArguments
    # ----------------------------
    training_args = build_training_args(args, use_4bit, fp16, bf16)

    # ----------------------------
    # Train (SFT)
    # ----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        data_collator=collator,
        #packing=args.packing,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # ----------------------------
    # Save adapter + tokenizer
    # ----------------------------
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n LoRA adapter saved to: {os.path.abspath(args.output_dir)}")
    print("   Look for adapter_config.json + adapter_model.safetensors")


if __name__ == "__main__":
    main()
