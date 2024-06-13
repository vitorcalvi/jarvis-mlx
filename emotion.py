import torch
import torchaudio
import pyaudio
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModelForSequenceClassification

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    device = torch.device("cpu")
    print("MPS device is not available, using CPU.")

# Load the Wav2Vec2.0 model and processor
wav2vec_model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).to(device)

# Function to record audio from the microphone
def record_audio(filename, duration, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = [stream.read(1024) for _ in range(0, int(sample_rate / 1024 * duration))]
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

# Process the waveform to get embeddings
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
inputs = inputs.to(device)

# Check the shape of the input tensor
print(f"Input shape: {inputs.shape}")

# Ensure the input tensor has the correct shape [batch_size, sequence_length]
inputs = inputs.squeeze()  # Remove any extra dimensions
inputs = inputs.unsqueeze(0)  # Add batch dimension

# Check the shape of the input tensor after adjustment
print(f"Adjusted input shape: {inputs.shape}")

with torch.no_grad():
    embeddings = wav2vec_model(inputs).last_hidden_state

# Use the mean of the embeddings across the time dimension
mean_embeddings = embeddings.mean(dim=1)

# Simple classifier example (this should be replaced with a real classifier)
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Assuming we have 7 emotion classes
num_emotion_classes = 7
classifier = SimpleClassifier(mean_embeddings.shape[-1], num_emotion_classes).to(device)

# Load your pre-trained classifier weights here
# classifier.load_state_dict(torch.load("path_to_your_classifier_weights.pth"))

# Perform inference
with torch.no_grad():
    emotion_logits = classifier(mean_embeddings)

# Get the predicted emotion
predicted_class_id = torch.argmax(emotion_logits, dim=-1).item()
# Assuming you have a mapping from class IDs to emotion labels
id2label = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
predicted_emotion = id2label[predicted_class_id]

print(f"Predicted emotion: {predicted_emotion}")
