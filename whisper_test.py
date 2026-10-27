import whisper

model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\josho\OneDrive\Documents\Sound Recordings\Recording.mp3")
print(result["text"])