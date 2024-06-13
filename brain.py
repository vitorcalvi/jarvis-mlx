import torch
from speechbrain.pretrained import EncoderDecoderASR

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    device = torch.device("cpu")
    print("MPS device is not available, using CPU.")

# Load the ASR model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="tmpdir")
asr_model = asr_model.to(device)

# Example audio data (replace with actual audio data)
audio_data = torch.randn(1, 16000).to(device)

# Perform speech recognition
transcription = asr_model.transcribe_batch(audio_data)
print(f"Transcription: {transcription}")
