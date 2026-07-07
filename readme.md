# Requirements
Make sure you have Ollama installed
Make sure the file qwen2.5-7b-instruct.Q4_K_M is download and on Bergamota's main folder
You can find the file at: https://drive.google.com/file/d/1MQsO7VJos4qekZFT4A4FvQC1jxI0Zupw/view?usp=sharing

# Install the model
ollama create literatura_gaucha_qwen -f Modelfile

# Pull ollama embebed-text
ollama pull nomic-embed-text

# Create a virtual environment named 'bergamota_env'
python -m venv bergamota_env

# Activate the environment
.\bergamota_env\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt

# Run Model
python bergamota.py
