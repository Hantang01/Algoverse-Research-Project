# Algoverse-Research-Project

Important: This project requires CUDA-enabled PyTorch and a local LLaMA model directory to run model inference. See the sections below for setup details.

## CUDA and Model Requirements

- CUDA-enabled GPU (recommended) with the appropriate NVIDIA drivers installed. You can check your GPU and driver with:

```powershell
nvidia-smi
```

- Install a CUDA-enabled PyTorch build (example for CUDA 12.1):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

- LLaMA model: place your model files in the project root under a folder named `my_llama_model/`. The code expects the model to be at `./my_llama_model` (this matches the default in `scripts/functions.py`). Example structure:

```
my_llama_model/
├── config.json
├── pytorch_model.bin or *.safetensors
├── tokenizer.json
└── other model files
```

You must ensure the model files are compatible with Hugging Face `transformers` (or the loading method used in `scripts/functions.py`).

## Quick verification

After installing the CUDA PyTorch build and placing your model in `my_llama_model/`, run a quick check:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)"
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('./my_llama_model'); print('Tokenizer loaded')"
```

If these commands succeed and `CUDA available: True` is printed, you're good to run `scripts/main.py`.
