import whisper
import pyaudio
import numpy as np
import keyboard
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model on the appropriate device
model = whisper.load_model("base").to("cuda")

# Audio stream setup
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # 1 channel
RATE = 16000  # 16kHz sampling rate
CHUNK = 1024  # 1024 samples per chunk
SILENCE_THRESHOLD = 500  # threshold for silence detection
SILENCE_CHUNKS = 30  # number of chunks of silence to trigger end of recording

# Initialize pyaudio
audio = pyaudio.PyAudio()

# Function to record audio and return as numpy array
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    silence_count = 0
    have_spoken = False

    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

        # Check if the audio is silent
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        if np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD:
            silence_count += 1
            have_spoken = True
        else:
            silence_count = 0

        # If silence is detected for a certain number of chunks, stop recording
        if silence_count > SILENCE_CHUNKS and have_spoken:
            break

        # If a key is pressed, stop recording
        if keyboard.is_pressed(hotkey = 'q'):
            print("Key pressed, stopping recording.")
            break

    stream.stop_stream()
    stream.close()

    # Convert the list of frames into a numpy array
    audio_data = np.concatenate(frames)
    # Convert int16 audio data to float32
    audio_data = audio_data.astype(np.float32) / 32768.0
    return audio_data

# Real-time speech-to-text transformation
try:
    while True:
        audio_data = record_audio()
        result = model.transcribe(audio_data)
        print("Transcription:", result["text"])

        # Break the loop if a key is pressed
        if keyboard.is_pressed(hotkey = 'q'):
            print("Exiting...")
            break
except KeyboardInterrupt:
    print("Interrupted by user.")

# Terminate pyaudio
audio.terminate()
