from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
import json
import time

# -------------------------
# Config
# -------------------------
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


MODELS = [
#    "microsoft/BioGPT-Large",
#    "stanford-crfm/BioMedLM",
#    "microsoft/BioGPT-Large-PubMedQA",
#    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
#    "Qwen/Qwen2.5-1.5B",
#    "Qwen/Qwen2.5-3B",
#    "Qwen/Qwen2.5-7B",
#    "deepseek-ai/deepseek-llm-7b-base",
#    "mistralai/Mistral-7B-v0.1",
    "microsoft/Phi-3-mini-4k-instruct" #,
#     "microsoft/Phi-4-reasoning-plus"
]

# config
GEN_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.0,
)

# seed
SEED = 42


def print_gpu_info():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU (this will be slower).")


def benchmark_model(model_name: str):
    """Benchmark one single model with stable decoding."""
    print(f"\n Loading {model_name}...")

    try:
        start_load = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # dtype: fp16 on GPU, fp32 on CPU 
        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if use_gpu else None
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_time = time.time() - start_load
        print(f"   Model loaded in {load_time:.1f}s")

        # pipeline with stable defaults
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            device_map="auto" if use_gpu else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GEN_CONFIG,
        )

        results = []

        for i, question in enumerate(QUESTIONS):
            print(f"   Q{i+1}: {question[:50]}...")

            prompt = f"Answer this clinical genetics question: {question}\n\nAnswer:"

            start_time = time.time()
            try:
                response = pipe(prompt)
                response_time = time.time() - start_time

                generated = response[0]["generated_text"]
                answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()

                results.append({
                    "model": model_name,
                    "question": question,
                    "answer": answer,
                    "response_time": round(response_time, 2),
                    "load_time": round(load_time, 2),
                    "success": True,
                    "gen_config": GEN_CONFIG
                })

                print(f"      Answered in {response_time:.1f}s")

            except Exception as e:
                print(f"      Error: {str(e)[:80]}...")
                results.append({
                    "model": model_name,
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "response_time": 0,
                    "load_time": round(load_time, 2),
                    "success": False,
                    "gen_config": GEN_CONFIG
                })

            time.sleep(0.5)

        # Clean up
        del model, tokenizer, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f" Failed to load {model_name}: {e}")
        return []


def main():
    print_gpu_info()
    set_seed(SEED)

    all_results = []
    print("Starting Clinical Genetics Benchmark with Transformers")
    print(f"Testing {len(MODELS)} models on {len(QUESTIONS)} questions")
    print("=" * 60)

    progress_path = "transformers_benchmark_progress.json"
    final_path = "transformers_benchmark_final.json"

    for model_name in MODELS:
        results = benchmark_model(model_name)
        all_results.extend(results)

        # Save progress
        with open(progress_path, "w") as f:
            json.dump(all_results, f, indent=2)

        successful = sum(1 for r in results if r["success"])
        print(f" Completed {model_name}: {successful}/{len(QUESTIONS)} questions")

    # Save final results
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\nFINAL SUMMARY")
    print("=" * 40)

    successful_models = 0
    total_answers = 0

    for model_name in MODELS:
        model_results = [r for r in all_results if r["model"] == model_name]
        successful = sum(1 for r in model_results if r["success"])
        total_answers += successful

        if model_results:
            avg_time = (
                sum(r["response_time"] for r in model_results if r["success"]) / max(successful, 1)
            )
            print(f" {model_name}:")
            print(f"    {successful}/{len(QUESTIONS)} answers")
            print(f"   ⏱️  Avg response: {avg_time:.1f}s")
            if successful > 0:
                successful_models += 1

    print(f"\n Overall: {successful_models}/{len(MODELS)} models successful")
    print(f" Total answers: {total_answers}/{(len(MODELS) * len(QUESTIONS))}")
    print(f" Results saved to {final_path}")


if __name__ == "__main__":
    main()
