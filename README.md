# PulseTrust (Python / PyQt6)
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py --demo      # try the HUD with synthetic data
    python main.py             # live: screen + speaker loopback
Hotkeys: Ctrl+Shift+D show/hide HUD, Ctrl+Shift+Q quit. Tune config.py (set CAPTURE_REGION to your call window for best accuracy).
Optional: put an ONNX anti-spoof model at models/aasist.onnx (see config.AUDIO_MODEL_PATH).
