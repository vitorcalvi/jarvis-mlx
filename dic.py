import pyaudio
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import threading
from huggingface_hub import login

# Replace 'YOUR_AUTH_TOKEN' with your actual Hugging Face token
AUTH_TOKEN = "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"

# Login to Hugging Face with the token and save it to git credential helper
login(token=AUTH_TOKEN, add_to_git_credential=True)

# Initialize the pre-trained diarization pipeline with authentication token
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=AUTH_TOKEN)

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Function to process audio chunks
def process_audio_chunk(chunk):
    # Convert chunk to numpy array
    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    audio_segment = Segment(0, len(audio_data) / RATE)
    # Process the audio segment
    diarization = pipeline(audio_segment)
    # Print the diarization result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# Thread to read and process audio chunks
def read_audio_stream():
    while True:
        chunk = stream.read(CHUNK)
        threading.Thread(target=process_audio_chunk, args=(chunk,)).start()

# Start reading audio stream
threading.Thread(target=read_audio_stream).start()
