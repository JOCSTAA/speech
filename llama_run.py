# filename: run_llama3.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "./llama3_model"
model_tokenizer = "./llama3_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode and move it to GPU
model.eval()
model.to('cuda')

# Function to generate text
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs.input_ids, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Hello, how can I assist you today?"
print(generate_text(prompt))