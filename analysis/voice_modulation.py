import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import logging
from typing import Dict, Any

class VoiceModulation:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, y: np.ndarray, sr: int, n_fft: int = 2048) -> Dict[str, Any]:
        try:
            self.logger.debug(f"Starting voice modulation analysis. Input shape: {y.shape}, Sample rate: {sr}")

            # Ensure the audio is long enough for analysis
            min_duration = 0.5  # Minimum 0.5 seconds
            if len(y) < int(sr * min_duration):
                self.logger.warning(f"Audio sample too short. Padding to {min_duration} seconds.")
                y = np.pad(y, (0, int(sr * min_duration) - len(y)))

            # Adjust n_fft if necessary
            n_fft = min(n_fft, len(y))
            hop_length = n_fft // 4

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # Filter out zero pitches and corresponding magnitudes
            valid_pitches = pitches[magnitudes > 0]
            valid_magnitudes = magnitudes[magnitudes > 0]
            
            if len(valid_pitches) == 0:
                self.logger.warning("No valid pitches detected.")
                return {"modulation_score": 0.5, "pitch_stats": {}, "energy_stats": {}}
            
            pitch = np.average(valid_pitches, weights=valid_magnitudes)
            energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

            pitch_stats = self._compute_stats(valid_pitches)
            energy_stats = self._compute_stats(energy)

            modulation_score = (pitch_stats["std"] + energy_stats["std"]) / 2

            self.logger.debug(f"Voice modulation analysis completed. Modulation score: {modulation_score}")

            return {
                "modulation_score": float(modulation_score),
                "pitch_stats": pitch_stats,
                "energy_stats": energy_stats
            }
        except Exception as e:
            self.logger.error(f"Error in VoiceModulation analysis: {str(e)}", exc_info=True)
            return {"modulation_score": 0.5, "pitch_stats": {}, "energy_stats": {}}

    def _compute_stats(self, data: np.ndarray) -> Dict[str, float]:
        if len(data) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0
            }
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "kurtosis": float(kurtosis(data) if len(data) > 1 else 0),
            "skewness": float(skew(data) if len(data) > 1 else 0)
        }