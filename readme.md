# Requirements
Make sure you have Ollama installed

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
