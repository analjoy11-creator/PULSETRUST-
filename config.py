from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
TEST_ASSETS_DIR = BASE_DIR / "test_assets"
 
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
 
# Hardware and Stream Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SAMPLES = 1024
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
 
# UI Overlay Dimensions & Configurations
OVERLAY_SIZE = (340, 460)
OVERLAY_MARGIN = 20
GLOBAL_HOTKEY = "ctrl+shift+l"   # show / hide HUD
RESET_HOTKEY = "alt+l"            # unlock + restart verification
 
# 4-Channel Forensic Thresholds
RPPG_BUFFER_LEN = 90          # 3 seconds @ 30 FPS
RPPG_BANDPASS = [0.75, 2.5]   # 45 to 150 BPM
EAR_BLINK_THRESHOLD = 0.20
SYNC_HISTORY_LEN = 45
 
# Fusion Engine Hyperparameters
ATTENTION_TAU = 0.28
TEMPORAL_ALPHA = 0.35
HIGH_THREAT_THRESHOLD = 70.0
SUSPICIOUS_THRESHOLD = 40.0
 