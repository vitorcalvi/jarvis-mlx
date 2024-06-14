import torch
import torchaudio
import pyaudio
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    device = torch.device("cpu")
    print("MPS device is not available, using CPU.")

# Load the Wav2Vec2.0 model and processor
wav2vec_model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name).to(device)

# Function to record audio from the microphone
def record_audio(filename, duration, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = []
    try:
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
    except Exception as e:
        print(f"An error occurred while recording: {e}")
    finally:
        print("Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Record audio from the microphone
audio_file_path = "recorded_audio.wav"
record_audio(audio_file_path, duration=5)  # Record for 5 seconds

# Load the recorded audio file
waveform, sample_rate = torchaudio.load(audio_file_path)

# Ensure the waveform is on the correct device
waveform = waveform.to(device)

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Process the waveform to get input features for Wav2Vec2
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
inputs = inputs.to(device)

# Ensure the input tensor has the correct shape [batch_size, sequence_length]
inputs = inputs.squeeze()  # Remove any extra dimensions
if len(inputs.shape) == 1:
    inputs = inputs.unsqueeze(0)  # Add batch dimension if missing

# Check the shape of the input tensor after adjustment
print(f"Adjusted input shape: {inputs.shape}")

# Perform inference with Wav2Vec2 model
with torch.no_grad():
    logits = wav2vec_model(inputs).logits

# Decode the logits to get the transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

transcribed_text = transcription[0]
print(f"Transcription: {transcribed_text}")

# Load the emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Predict the emotion of the transcribed text
emotion_predictions = emotion_classifier(transcribed_text)

# Print the emotion predictions
print("Emotion Predictions:")
for emotion in emotion_predictions[0]:
    print(f"{emotion['label']}: {emotion['score']:.4f}")
