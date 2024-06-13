from huggingface_hub import hf_hub_download

# Hugging Face access token
HF_TOKEN = "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"  # Replace with your actual token

# Download the config.yaml file
config_path = hf_hub_download(
    repo_id="pyannote/speaker-diarization",
    filename="config.yaml",
    use_auth_token=HF_TOKEN
)

print(f"Config file downloaded to: {config_path}")
