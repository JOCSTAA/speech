import sounddevice as sd
import numpy as np
import queue
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nemo.collections.asr as nemo_asr

# --- Load ASR Model (NeMo) ---
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_quartznet15x5")

# --- Load Mistral Model ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# --- Audio Stream Config ---
sample_rate = 16000
block_size = 2048
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def transcribe_audio(audio):
    text = asr_model.transcribe([audio])[0]
    return text.strip()

def query_mistral(prompt, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Return only the new response, not the prompt
    return generated[len(prompt):].strip()

# --- Main Loop ---
with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=block_size):
    print("🎤 Speak now (Ctrl+C to stop)...")
    buffer = []

    try:
        while True:
            audio_chunk = q.get()
            buffer.extend(audio_chunk.flatten())

            if len(buffer) > sample_rate * 2:  # ~2 seconds of audio
                audio_array = np.array(buffer, dtype=np.float32)
                transcription = transcribe_audio(audio_array)
                print(f"\n🗣️ You said: {transcription}")

                if transcription:
                    prompt = f"<s>[INST] {transcription} [/INST]"
                    print("🤖 Mistral is thinking...")
                    reply = query_mistral(prompt)
                    print(f"💬 Mistral: {reply}")

                buffer = []

    except KeyboardInterrupt:
        print("\n🎤 Session ended.")
