import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline
import logging
import numpy as np
from typing import Dict, Any
import librosa

class EmotionalExpression:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_classifier = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

    def analyze_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        try:
            # Ensure the audio is long enough (at least 1 second)
            if len(y) < sr:
                y = np.pad(y, (0, sr - len(y)), mode='constant')
            
            # Resample to 16kHz if necessary
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Extract features
            features = self.audio_feature_extractor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_classifier(**features)
            
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            emotion = self.audio_classifier.config.id2label[scores.argmax().item()]
            confidence = float(scores.max().item())
            
            return {"emotion": emotion, "confidence": confidence}
        except Exception as e:
            logging.error(f"Error in EmotionalExpression audio analysis: {str(e)}")
            return {"emotion": "unknown", "confidence": 0.0}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        try:
            sentiment = self.nlp(text)[0]
            return {"emotion": sentiment["label"], "score": float(sentiment["score"])}
        except Exception as e:
            logging.error(f"Error in EmotionalExpression text analysis: {str(e)}")
            return {"emotion": "unknown", "score": 0.0}