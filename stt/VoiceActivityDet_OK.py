import time
import wave
import pyaudio
import webrtcvad
import contextlib
import collections
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Constants
RATE = 16000
CHUNK = 320  # Increased chunk size
CHANNELS = 1
FORMAT = pyaudio.paInt16
SENSITIVITY = 0.5  # Adjusted sensitivity in seconds
END_SPEECH_BUFFER = 1.0  # Buffer period after detecting end of speech in seconds
MIN_SPEECH_DURATION = 0.5  # Minimum duration of speech to be considered valid in seconds

# Hugging Face model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

class VADDetector:
    def __init__(self, on_speech_start, on_speech_end, sensitivity=SENSITIVITY):
        self.sample_rate = RATE
        self.interval_size = 20  # Increased interval size in ms
        self.sensitivity = sensitivity
        self.block_size = int(self.sample_rate * self.interval_size / 1000)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Less aggressive mode
        self.frame_history = [False]
        self.block_since_last_spoke = 0
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.voiced_frames = collections.deque(maxlen=1000)
        self.speech_started = False
        self.end_speech_buffer_blocks = int(END_SPEECH_BUFFER * 1000 / self.interval_size)
        self.min_speech_duration_blocks = int(MIN_SPEECH_DURATION * 1000 / self.interval_size)
        self.speech_start_time = None

    def voice_activity_detection(self, audio_data):
        return self.vad.is_speech(audio_data, self.sample_rate)

    def audio_callback(self, indata, frames, time, status):
        detection = self.voice_activity_detection(indata)
        if detection:
            if not self.speech_started:
                self.speech_start_time = time
                self.on_speech_start()
                self.speech_started = True
            self.voiced_frames.append(indata)
            self.block_since_last_spoke = 0
        else:
            if self.speech_started:
                self.block_since_last_spoke += 1
                if self.block_since_last_spoke > self.end_speech_buffer_blocks:
                    # Check if the speech duration is long enough
                    speech_duration = time - self.speech_start_time
                    if speech_duration >= MIN_SPEECH_DURATION:
                        if self.voiced_frames:
                            audio_data = b"".join(self.voiced_frames)
                            self.on_speech_end(np.frombuffer(audio_data, dtype=np.int16))
                    self.voiced_frames.clear()
                    self.speech_started = False
        self.frame_history.append(detection)

    def start_listening(self):
        stream = pyaudio.PyAudio().open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_callback(data, CHUNK, time.time(), None)
            except Exception as e:
                print(e)
                break

def on_speech_start():
    print("Speech started")

def on_speech_end(data):
    print("Speech ended")
    # Store audio data for debugging purposes
    global debug_audio_data
    debug_audio_data = data

def transcribe_audio_data(audio_data):
    # Convert data to float32
    data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    input_values = processor(data, sampling_rate=RATE, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(f"Transcription: {transcription[0]}")

if __name__ == "__main__":
    vad = VADDetector(on_speech_start, on_speech_end)
    vad.start_listening()
    transcribe_audio_data()

    # For debugging purposes, call transcribe_audio_data with the stored audio data
    if 'debug_audio_data' in globals():
        transcribe_audio_data(debug_audio_data)
