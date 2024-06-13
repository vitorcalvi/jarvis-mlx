import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    device = torch.device("cpu")
    print("MPS device is not available, using CPU.")

# Load the pre-trained Wav2Vec 2.0 model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

# Load an audio file
audio_file_path = "wav2.wav"
waveform, sample_rate = torchaudio.load(audio_file_path)

# Ensure the waveform is on the correct device
waveform = waveform.to(device)

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Process the waveform and get the logits
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
inputs = inputs.to(device)
with torch.no_grad():
    logits = model(inputs).logits

# Decode the logits to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(f"Transcription: {transcription[0]}")
