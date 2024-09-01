import numpy as np
import librosa
import logging
from typing import Dict, Any

class FluencyIntonation:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, y: np.ndarray, sr: int, n_fft: int = 2048) -> Dict[str, Any]:
        try:
            self.logger.debug(f"Starting fluency and intonation analysis. Input shape: {y.shape}, Sample rate: {sr}, n_fft: {n_fft}")

            # Ensure the audio is long enough for analysis
            if len(y) < sr // 2:  # At least 0.5 seconds of audio
                self.logger.warning("Audio sample too short. Padding with zeros.")
                y = np.pad(y, (0, sr // 2 - len(y)))

            # Split audio into speech segments
            intervals = librosa.effects.split(y, top_db=20)
            total_speech = sum(i[1] - i[0] for i in intervals) / sr
            total_duration = len(y) / sr
            speech_ratio = total_speech / total_duration

            # Detect beats for speech rate estimation
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=n_fft//4)
            speech_rate = len(beats) / total_duration * 60

            # Calculate fluency score
            fluency_score = speech_ratio * min(speech_rate / 150, 1)

            # Calculate pitch variation for intonation
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//4)
            pitch_variation = np.std(pitches[magnitudes > np.median(magnitudes)])

            self.logger.debug(f"Fluency and intonation analysis completed. Fluency score: {fluency_score}, Pitch variation: {pitch_variation}")

            return {
                "fluency_score": float(fluency_score),
                "speech_ratio": float(speech_ratio),
                "speech_rate": float(speech_rate),
                "pitch_variation": float(pitch_variation)
            }
        except Exception as e:
            self.logger.error(f"Error in FluencyIntonation analysis: {str(e)}", exc_info=True)
            return {"fluency_score": 0.5, "speech_ratio": 0.5, "speech_rate": 0, "pitch_variation": 0}