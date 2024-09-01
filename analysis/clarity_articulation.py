import numpy as np
import librosa
import logging
from typing import Dict

class ClarityArticulation:
    def analyze(self, y: np.ndarray, sr: int, n_fft: int = 2048) -> Dict[str, float]:
        try:
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
            clarity_score = float(np.mean(cent) / (sr / 2))
            return {
                "clarity_score": clarity_score,
                "spectral_centroid_mean": float(np.mean(cent)),
                "spectral_centroid_std": float(np.std(cent))
            }
        except Exception as e:
            logging.error(f"Error in ClarityArticulation analysis: {str(e)}")
            return {"clarity_score": 0.5, "spectral_centroid_mean": 0, "spectral_centroid_std": 0}