import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def evaluate_transformer(val_texts, config):
    model_name = config['inference']['transformer_model']
    device_id = 0 if torch.cuda.is_available() else -1
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(model_device)
    
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device_id
    )
    rouge_metric = evaluate.load("rouge")
    
    predictions, references, losses = [], [], []

    for text in val_texts:
        words = str(text).split()
        if len(words) < 8: 
            continue
        
        split_idx = int(len(words) * 0.75)
        prompt, reference = " ".join(words[:split_idx]), " ".join(words[split_idx:])

        out = generator(
            prompt, 
            max_new_tokens=20,
            num_return_sequences=1, 
            do_sample=False, 
            temperature=None, 
            top_p=None,
            pad_token_id=tokenizer.eos_token_id, 
        )[0]

        full_gen = out['generated_text']
        predictions.append(full_gen[len(prompt):].strip())
        references.append(reference)

        inputs = tokenizer(full_gen, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    res = rouge_metric.compute(predictions=predictions, references=references)
    ppl = np.exp(np.mean(losses)) if losses else 0

    print(f"\n[Model: {model_name}]")
    print(f"PPL: {ppl:.4f} | ROUGE-1: {res['rouge1']:.4f} | ROUGE-2: {res['rouge2']:.4f}\n")

    return {
        "ppl": ppl, 
        "rouge1": res['rouge1'], 
        "rouge2": res['rouge2']
    }
