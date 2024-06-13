import warnings
import time
import os
import threading
import argparse
from queue import Queue
import numpy as np
import librosa
import sounddevice as sd
from playsound import playsound
from pydantic import BaseModel
from melo.api import TTS
from stt.VoiceActivityDet import VADDetector
from mlx_lm import load, generate
from stt.whisper.transcribe import FastTranscriber

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

def parse_args():
    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument('--agent', type=str, default='python teacher', help='The agent type')
    args = parser.parse_args()
    return args

args = parse_args()
AGENT = args.agent if args.agent else 'python teacher'

# Prompts for the assistant
MASTER_PROMPT = f"You are a helpful and friendly {AGENT}. Respond to the user's input in no more than one great sentence, ensuring complete ideas. Provide only the dialogue in your response."
SUB_MASTER_PROMPT = f"Hello! I'm your {AGENT}"

class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, startListening=True, history=None):
        self.history = history or []
        self.listening = False
        self.vad = VADDetector(lambda: None, self.on_speech_end, sensitivity=0.5)
        self.vad_data = Queue()
        self.tts = TTS(language="EN_NEWEST", device="mps")
        self.stt = FastTranscriber("mlx-community/whisper-large-v3-mlx-4bit")
        self.model, self.tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-8bit")
        self.greet()

        if startListening:
            self.toggle_listening()
            self.start_listening()
            threading.Thread(target=self.transcription_loop, daemon=True).start()

    def greet(self):
        os.system('cls||clear')
        print(SUB_MASTER_PROMPT)
        self.speak(SUB_MASTER_PROMPT)
        self.listening = False

    def start_listening(self):
        threading.Thread(target=self.vad.start_listening, daemon=True).start()

    def toggle_listening(self):
        if not self.listening:
            playsound("beep.mp3")
            print("\033[36mListening...\033[0m")
        while not self.vad_data.empty():
            self.vad_data.get()
        self.listening = not self.listening

    def on_speech_end(self, data):
        if data.size > 0:
            self.vad_data.put(data)

    def add_to_history(self, content: str, role: str):
        color = "\033[32m" if role == "user" else "\033[33m"
        print(f"{color}{content}\033[0m")
        if role == "user":
            content = f"{MASTER_PROMPT}\n\n{content}"
        self.history.append(ChatMLMessage(content=content, role=role))

    def get_history_as_string(self):
        return "".join(f"<|{message.role}|>{message.content}<|end|>\n" for message in self.history)

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()
                if self.listening and len(data) > 12000:  # Ensure the data length is sufficient
                    self.toggle_listening()
                    try:
                        transcribed = self.stt.transcribe(data, language="en")
                        print(f"Transcribed: {transcribed}")
                        self.add_to_history(transcribed["text"], "user")
                        history = self.get_history_as_string()
                        response = generate(self.model, self.tokenizer, prompt=history + "\n<|assistant|>", verbose=False)
                        response = response.split("<|assistant|>")[0].split("<|end|>")[0].strip()
                        self.add_to_history(response, "assistant")
                        self.speak(response)
                    except Exception as e:
                        print(f"Error during transcription or generation: {e}")
                        self.toggle_listening()

    def speak(self, text):
        try:
            data = self.tts.tts_to_file(text, self.tts.hps.data.spk2id["EN-Newest"], speed=1, quiet=True, sdp_ratio=0.5)
            trimmed_audio, _ = librosa.effects.trim(data, top_db=20)
            
            # Add 0.5 seconds of silence at the beginning and end
            silence_duration = 0.5
            sample_rate = 44100
            silence = np.zeros(int(silence_duration * sample_rate))
            padded_audio = np.concatenate([silence, trimmed_audio, silence])
            
            sd.play(padded_audio, sample_rate, blocking=True)
            sd.wait()  # Ensure playback is finished
            self.toggle_listening()
        except Exception as e:
            print(f"Error during TTS or playback: {e}")
            self.toggle_listening()

if __name__ == "__main__":
    client = Client(startListening=True, history=[])
    # Keep the main thread alive
    while True:
        time.sleep(1)
