from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import time

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# questions to test
QUESTIONS = [
"""
You are an expert assistant in human genetics and genomics.Answer in a clinically useful, structured way. Be accurate and do not invent facts. If evidence is insufficient, say so.You must Comment every variant, check for frequency, Clinvar class, Mode of Inheritence and CADD.You Must report the Allelic balance and the Frequency of each variant, if no frequency mentioned in the input table, it is 0.
Here is a table contains 5 variants. Wchich variant cause Obesity?                                                                                                                                                                                                           
Variant 1: Variant=chr19:40730680_1 insG, Chromosome=chr19, Position=40730681, RS_ID=NA, Ref_seq=NA, Var_seq=G, Type=Insertion, HGVS=NM_024877.4:c.305dup p.(Val103CysfsTer25), Zygosity=Heterozygous, Gene=CCNP, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinV\
ar_class=NA, Allelic_balance=0.5417, Frequency=NA, CADD_score=NA                                                                                                                                                                                                                
Variant 2: Variant=chr16:56904031 C>T, Chromosome=chr16, Position=56904031, RS_ID=rs28936388, Ref_seq=C, Var_seq=T, Type=SNV, HGVS=NM_001126108.2:c.625C>T p.(Arg209Trp), Zygosity=Heterozygous, Gene=SLC12A3, OMIM_phenotype=Gitelman syndrome, OMIM_inheritance=Autosomal rec\
essive, Inheritance=AR, ClinVar_class=Pathogenic, Allelic_balance=0.5393, Frequency=1.19E-05, CADD_score=26                                                                                                                                                                     
Variant 3: Variant=chr12:117962872 G>A, Chromosome=chr12, Position=117962872, RS_ID=rs56178407, Ref_seq=G, Var_seq=A, Type=SNV, HGVS=NM_173598.6:c.2004C>T p.(Ile668=), Zygosity=Heterozygous, Gene=KSR2, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinVar_class\
=Benign, Allelic_balance=0.3947, Frequency=0.00689813, CADD_score=5.871                                                                                                                                                                                                         
Variant 4: Variant=chr3:49690627 G>A, Chromosome=chr3, Position=49690627, RS_ID=rs35762866, Ref_seq=G, Var_seq=A, Type=SNV, HGVS=NM_003458.4:c.3638G>A p.(Gly1213Asp), Zygosity=Heterozygous, Gene=BSN, OMIM_phenotype=NA, OMIM_inheritance=NA, Inheritance=NA, ClinVar_class=B\
enign, Allelic_balance=0.5556, Frequency=0.0813352, CADD_score=12.92                                                                                                                                                                                                            
Variant 5: Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, RS_ID=NA, Ref_seq=T, Var_seq=C, Type=SNV, HGVS=NM_005068.3:c.744-2A>G p.?, Zygosity=Heterozygous, Gene=SIM1, OMIM_phenotype=Obesity, OMIM_inheritance=NA, Inheritance=AD, ClinVar_class=NA, Allelic_bala\
nce=0.383, Frequency=NA, CADD_score=34                                                                                                                                                                                                                                         
"""
]

# Small models that work well with transformers
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
    "microsoft/Phi-3-mini-4k-instruct"
#     "microsoft/Phi-4-reasoning-plus"
]

def benchmark_model(model_name):
    """Benchmark a single model"""
    print(f"\n Loading {model_name}...")
    
    try:
        start_load = time.time()
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Parameter Jing precision/memory
            device_map="auto" #use CPU if no GPU 
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        load_time = time.time() - start_load
        print(f"    Model loaded in {load_time:.1f}s")
        
        # pipeline
        # The less halluci / max token 1024
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        results = []
        
        for i, question in enumerate(QUESTIONS):
            print(f"   Q{i+1}: {question[:50]}...")
            
            prompt = f"Answer this clinical genetics question: {question}\n\nAnswer:"
            
            start_time = time.time()
            try:
                response = pipe(prompt)
                response_time = time.time() - start_time
                
                answer = response[0]['generated_text'].replace(prompt, "").strip()
                
                results.append({
                    'model': model_name,
                    'question': question,
                    'answer': answer,
                    'response_time': round(response_time, 2),
                    'load_time': round(load_time, 2),
                    'success': True
                })
                
                print(f"       Answered in {response_time:.1f}s")
                
            except Exception as e:
                print(f"       Error: {str(e)[:50]}...")
                results.append({
                    'model': model_name,
                    'question': question,
                    'answer': f"ERROR: {str(e)}",
                    'response_time': 0,
                    'load_time': round(load_time, 2),
                    'success': False
                })
            
            # Small delay between questions
            time.sleep(1)
        
        # Clean up GPU memory
        del model, tokenizer, pipe
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f" Failed to load {model_name}: {e}")
        return []

def main():
    all_results = []
    
    print("Starting Clinical Genetics Benchmark with Transformers")
    print(f"Testing {len(MODELS)} models on {len(QUESTIONS)} questions")
    print("=" * 60)
    
    for model in MODELS:
        results = benchmark_model(model)
        all_results.extend(results)
        
        # Save progress after each model
        with open('progress_phi3.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        successful = len([r for r in results if r['success']])
        print(f" Completed {model}: {successful}/{len(QUESTIONS)} questions")
    
    # Save final results
    with open('20Questions_genetic_knowdege_phi3.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n FINAL SUMMARY")
    print("=" * 40)
    
    successful_models = 0
    total_answers = 0
    
    for model in MODELS:
        model_results = [r for r in all_results if r['model'] == model]
        successful = len([r for r in model_results if r['success']])
        total_answers += successful
        
        if model_results:
            avg_time = sum(r['response_time'] for r in model_results if r['success']) / max(successful, 1)
            print(f" {model}:")
            print(f"    {successful}/{len(QUESTIONS)} answers")
            print(f"     Avg response: {avg_time:.1f}s")
            
            if successful > 0:
                successful_models += 1
    
    print(f"\n Overall: {successful_models}/{len(MODELS)} models successful")
    print(f" Total answers: {total_answers}/{(len(MODELS) * len(QUESTIONS))}")
    print(" Results saved to transformers_benchmark_final.json")

if __name__ == "__main__":
    main()
