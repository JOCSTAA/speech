from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "mistralai/Mistral-7B-v0.1"

# Quantization configuration
def load_quantized_model(model_name, dtype="int8"):
    print(f"Loading {model_name} with quantization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        load_in_8bit=(dtype == "int8"),  # Set to True for 8-bit quantization
        load_in_4bit=(dtype == "int4"),  # Optional for 4-bit (requires special config)
        torch_dtype=torch.float16 if dtype == "fp16" else torch.float32
    )
    return model, tokenizer

# Load Mistral 7B in 8-bit
mistral_model, mistral_tokenizer = load_quantized_model("mistral", dtype="int8")

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
mistral_prompt = "Hello my name is Josh, I will be working with you today"

# Generate text
print("Mistral Output:")
print(generate_text(mistral_model, mistral_tokenizer, mistral_prompt))


# LOAD IN SPEECH TO TEXT MODEL (whisper maybe)
# docker run -d \
#   --name=code-server \
#   --restart=unless-stopped \
#   -e PUID=1000 \
#   -e PGID=1000 \
#   -e TZ="America/Los_Angeles" \
#   -e PASSWORD="codeserver" \
#   -p 8443:8443 \
#   -v ~/docker/code-server/config:/config \
#   -v ~/docker/code-server/projects:/config/workspace \
#   --gpus all \
#   linuxserver/code-server
