from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose the model and tokenizer
models = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama2": "meta-llama/Llama-2-7b-hf"
}

# Quantization configuration
def load_quantized_model(model_name, dtype="int8"):
    print(f"Loading {model_name} with quantization...")
    tokenizer = AutoTokenizer.from_pretrained(models[model_name], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        models[model_name],
        device_map="auto",
        load_in_8bit=(dtype == "int8"),  # Set to True for 8-bit quantization
        load_in_4bit=(dtype == "int4"),  # Optional for 4-bit (requires special config)
        torch_dtype=torch.float16 if dtype == "fp16" else torch.float32
    )
    return model, tokenizer

# Load Mistral 7B in 8-bit
mistral_model, mistral_tokenizer = load_quantized_model("mistral", dtype="int8")

# Load LLaMA 2 7B in 8-bit
llama2_model, llama2_tokenizer = load_quantized_model("llama2", dtype="int8")

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,  # Adjust for creativity
            top_p=0.9,       # Top-p sampling for better coherence
            repetition_penalty=1.2  # Penalize repetition
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompts
mistral_prompt = "Once upon a time in a distant land, there lived a"
llama2_prompt = "In a world powered by advanced AI, humans decided to"

# Generate text
print("Mistral Output:")
print(generate_text(mistral_model, mistral_tokenizer, mistral_prompt))

print("\nLLaMA 2 Output:")
print(generate_text(llama2_model, llama2_tokenizer, llama2_prompt))
