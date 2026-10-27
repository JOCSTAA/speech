# import sounddevice as sd
# import queue
# import nemo.collections.asr as nemo_asr
# import numpy as np

# # Load pre-trained model
# asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
#     model_name="stt_en_quartznet15x5", map_location="cuda"
# )
# asr_model = asr_model.to(torch.float16)

# # Audio config
# sample_rate = 16000  # QuartzNet expects 16 kHz
# block_size = 2048
# q = queue.Queue()

# def callback(indata, frames, time, status):
#     if status:
#         print(status)
#     q.put(indata.copy())

# # Start audio stream
# with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=block_size):
#     print("Listening (Ctrl+C to stop)...")
#     buffer = []

#     try:
#         while True:
#             audio_chunk = q.get()
#             buffer.extend(audio_chunk.flatten())

#             if len(buffer) > sample_rate * 2:  # Every ~2 seconds
#                 audio_array = np.array(buffer, dtype=np.float32)
#                 text = asr_model.transcribe([audio_array])[0]
#                 print("🗣️  Recognized:", text)
#                 buffer = []  # reset buffer

#     except KeyboardInterrupt:
#         print("\nStopped.")

import torch
print(torch.__version__)
print(torch.version.cuda)