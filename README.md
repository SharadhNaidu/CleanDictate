# CleanDictate

Intelligent Dictation Engine with real-time speech-to-text, filler removal, and dual tone transformation (Professional & Casual).

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- Windows 10/11

## Installation

```powershell
# Clone the repository
git clone https://github.com/SharadhNaidu/CleanDictate.git
cd CleanDictate

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install faster-whisper transformers spacy pyaudio pynput scipy

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Run

```powershell
python cleandictate.py
```

## Hotkeys

| Key | Action |
|-----|--------|
| F9  | Start Recording |
| F10 | Stop Recording |
| F11 | Copy Output |
| ESC | Quit |

## Modes

- **Live**: Real-time VAD, instant typing
- **Document**: Record all, process on stop

## Tones

- `neutral` - No style change
- `formal` - Professional business tone
- `casual` - Friendly conversational tone
- `concise` - Brief and direct
- `dual` - Both formal & casual output
