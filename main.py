import sys
import time
import threading
import keyboard
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import QApplication
 
from config import GLOBAL_HOTKEY, RESET_HOTKEY, TARGET_FPS
from core.screen_capture import ScreenCaptureEngine
from core.audio_capture import AudioLoopbackCapture
from engines.rppg_engine import RPPGEngine
from engines.sync_engine import AudioVisualSyncEngine
from engines.landmark_engine import LandmarkEngine
from engines.audio_spoof_engine import AudioSpoofEngine
from fusion.fusion_engine import QualityGatedAttentionFusion
from ui.overlay_window import OverlayWindow, AssistantBridge
 
reset_event = threading.Event()
 
def pipeline_worker(bridge: AssistantBridge):
    screen_cap = ScreenCaptureEngine()
    rppg = RPPGEngine()
    sync = AudioVisualSyncEngine()
    landmarks_engine = LandmarkEngine()
    audio_spoof = AudioSpoofEngine()
    fusion = QualityGatedAttentionFusion()
 
    latest_pcm = [np.zeros(1024, dtype=np.float32)]
 
    def audio_sink(chunk: np.ndarray):
        latest_pcm[0] = chunk
 
    audio_cap = AudioLoopbackCapture(callback=audio_sink)
    audio_cap.start()
 
    mp_mesh = mp.solutions.face_mesh
    with mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            t0 = time.time()
 
            # alt+L: wipe all rolling history so verification restarts from zero
            if reset_event.is_set():
                reset_event.clear()
                rppg = RPPGEngine()
                sync = AudioVisualSyncEngine()
                landmarks_engine = LandmarkEngine()
                fusion = QualityGatedAttentionFusion()
 
            frame = screen_cap.grab_frame()
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
 
            # Default safe scores if no face is detected
            rppg_r, rppg_q = 0.1, 0.1
            sync_r, sync_q = 0.1, 0.1
            lm_r, lm_q = 0.1, 0.1
 
            # Audio analysis runs continuously regardless of face detection
            pcm_data = latest_pcm[0]
            audio_rms = float(np.sqrt(np.mean(pcm_data ** 2)))
            aud_r, aud_q = audio_spoof.analyze_audio_chunk(pcm_data)
 
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
 
                # 1. Lip Sync Aperture Correlation
                upper = np.array([lms[13].x * w, lms[13].y * h])
                lower = np.array([lms[14].x * w, lms[14].y * h])
                lip_dist = np.linalg.norm(upper - lower)
                sync_r, sync_q = sync.update(lip_dist, audio_rms)
 
                # 2. Eye Aspect Ratio (Blink Jitter)
                ear = landmarks_engine.compute_ear(lms, w, h)
                lm_r, lm_q = landmarks_engine.update(ear)
 
                # 3. Dynamic Forehead Crop for rPPG
                cx, cy = int(lms[10].x * w), int(lms[10].y * h)
                x1, y1 = max(0, cx - 25), max(0, cy - 25)
                forehead = rgb[y1:y1+50, x1:x1+50]
                rppg_r, rppg_q = rppg.extract_pulse_risk(forehead)
 
            if not results.multi_face_landmarks:
                # No face on screen -> no verdict (never report "authentic" on an empty window)
                fused = {
                    "risk_percentage": 10.0,
                    "status": "INCONCLUSIVE - NO FACE",
                    "primary_driver": "No face detected on screen",
                    "weights": [0.25, 0.25, 0.25, 0.25],
                }
            else:
                # Fuse the 4 physical modalities
                fused = fusion.fuse(
                    scores=[rppg_r, sync_r, lm_r, aud_r],
                    reliabilities=[rppg_q, sync_q, lm_q, aud_q]
                )
 
            bridge.telemetry_update.emit(fused)
 
            # Frame pacing
            dt = time.time() - t0
            time.sleep(max(0, (1.0 / TARGET_FPS) - dt))
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    bridge = AssistantBridge()
    window = OverlayWindow(bridge)
    window.show()
 
    keyboard.add_hotkey(GLOBAL_HOTKEY, lambda: bridge.toggle_visibility.emit())
    keyboard.add_hotkey(RESET_HOTKEY, lambda: (reset_event.set(), bridge.reset_analysis.emit()))
 
    t = threading.Thread(target=pipeline_worker, args=(bridge,), daemon=True)
    t.start()
 
    sys.exit(app.exec())