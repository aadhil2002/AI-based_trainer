import os
import torch
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoFeatureExtractor, AutoModelForAudioClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    WhisperProcessor, WhisperForConditionalGeneration
)
import numpy as np
import logging
from tqdm import tqdm
import pickle
import json
import librosa

# Disable symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Set logging level for transformers
import transformers
transformers.logging.set_verbosity_error()

class VideoProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)
        self.image_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)

    def process(self, visual_data):
        try:
            results = {
                "engagement": [],
                "confidence": [],
                "facial_expressions": [],
                "visual_features": []
            }

            for frame in tqdm(visual_data, desc="Processing visual", leave=False):
                # Placeholder for engagement, confidence, and facial expressions analysis
                results["engagement"].append(float(np.random.rand()))
                results["confidence"].append(float(np.random.rand()))
                results["facial_expressions"].append([float(x) for x in np.random.rand(5)])  # Placeholder for 5 facial expressions
                results["visual_features"].append(self.extract_visual_features(frame))

            return results
        except Exception as e:
            logging.error(f"Error in visual processing: {str(e)}")
            return {}

    def extract_visual_features(self, frame: list) -> list:
        try:
            frame_np = np.array(frame)
            inputs = self.image_processor(images=frame_np, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.image_model(**inputs)
            return outputs.logits.cpu().numpy().tolist()[0]
        except Exception as e:
            logging.error(f"Error in visual feature extraction: {str(e)}")
            return [0.0] * self.image_model.config.num_labels

class AudioProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)

    def process(self, audio_data):
        try:
            results = {
                "voice_modulation": [],
                "fluency": [],
                "clarity": [],
                "emotional_expression": [],
                "audio_features": [],
                "transcription": []
            }

            # Convert mel spectrogram back to audio signal
            mel_spec = np.array(audio_data)
            audio_signal = librosa.feature.inverse.mel_to_audio(mel_spec)

            # Placeholder for voice modulation, fluency, clarity, and emotional expression analysis
            num_chunks = len(audio_data)
            results["voice_modulation"] = [float(x) for x in np.random.rand(num_chunks)]
            results["fluency"] = [float(x) for x in np.random.rand(num_chunks)]
            results["clarity"] = [float(x) for x in np.random.rand(num_chunks)]
            results["emotional_expression"] = [[float(y) for y in x] for x in np.random.rand(num_chunks, 5)]  # Placeholder for 5 emotional expressions
            results["audio_features"] = self.extract_audio_features(audio_signal)
            results["transcription"] = self.transcribe_audio(audio_signal)

            return results
        except Exception as e:
            logging.error(f"Error in audio processing: {str(e)}")
            return {}

    def extract_audio_features(self, audio_data: list) -> list:
        try:
            audio_np = np.array(audio_data)

            # Reshape the audio data if necessary
            if len(audio_np.shape) == 3:  # (batch, channels, time)
                audio_np = audio_np.squeeze(0)  # Remove batch dimension
            if len(audio_np.shape) == 2:  # (channels, time)
                audio_np = np.mean(audio_np, axis=0)  # Convert to mono

            # Ensure the audio length is sufficient
            min_length = 16000  # 1 second at 16kHz
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)))

            inputs = self.audio_feature_extractor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt",
                max_length=16000,  # 1 second max
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.audio_model(**inputs)
            return outputs.logits.cpu().numpy().tolist()[0]
        except Exception as e:
            logging.error(f"Error in audio feature extraction: {str(e)}")
            return [0.0] * self.audio_model.config.num_labels

    def transcribe_audio(self, audio_data: list) -> str:
        try:
            audio_np = np.array(audio_data)
            input_features = self.whisper_processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features, language='en')  # Force English output
            
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0]
        except Exception as e:
            logging.error(f"Error in audio transcription: {str(e)}")
            return ""

class TextProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.text_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base").to(self.device)

    def process(self, text_data):
        try:
            results = {
                "phrasing_language": float(np.random.rand()),  # Placeholder for phrasing language analysis
                "emotional_expression": self.extract_emotions(text_data),
                "text_features": self.extract_text_features(text_data)
            }

            return results
        except Exception as e:
            logging.error(f"Error in text processing: {str(e)}")
            return {}

    def extract_emotions(self, text: list) -> list:
        try:
            inputs = self.text_tokenizer(" ".join(text), return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            return outputs.logits.cpu().numpy().tolist()[0]
        except Exception as e:
            logging.error(f"Error in emotion extraction: {str(e)}")
            return [0.0] * self.text_model.config.num_labels

    def extract_text_features(self, text: list) -> list:
        try:
            inputs = self.text_tokenizer(" ".join(text), return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            return outputs.logits.cpu().numpy().tolist()[0]
        except Exception as e:
            logging.error(f"Error in text feature extraction: {str(e)}")
            return [0.0] * self.text_model.config.num_labels

class MultimodalFeaturePipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.video_processor = VideoProcessor(self.device)
        self.audio_processor = AudioProcessor(self.device)
        self.text_processor = TextProcessor(self.device)

    def process_interview(self, preprocessed_data) -> dict:
        results = {}
        for filename, data in tqdm(preprocessed_data.items(), desc="Processing interviews"):
            visual_data, visual_label, audio_data, audio_label, text_data, text_label, overall_label = data

            results[filename] = {
                "visual": self.video_processor.process(visual_data),
                "audio": self.audio_processor.process(audio_data),
                "text": self.text_processor.process(text_data),
                "labels": {
                    "visual": visual_label,
                    "audio": audio_label,
                    "text": text_label,
                    "overall": overall_label
                }
            }
        return results

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing Multimodal Feature Pipeline")
    pipeline = MultimodalFeaturePipeline()

    # Define custom directories
    input_directory = r'D:\AI-based_trainer\data'  # Replace with your input directory path
    output_directory = r'D:\AI-based_trainer\data'  # Replace with your output directory path

    try:
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Load preprocessed data from the custom input directory
        input_file_path = os.path.join(input_directory, "preprocessed_data.pkl")
        logging.info(f"Loading preprocessed data from {input_file_path}")

        if not os.path.exists(input_file_path):
            logging.error(f"Input file not found: {input_file_path}")
            return

        with open(input_file_path, "rb") as f:
            preprocessed_data = pickle.load(f)

        logging.info("Processing interview")
        results = pipeline.process_interview(preprocessed_data)

        logging.info("Analysis complete. Saving results.")

        # Save results to the custom output directory
        output_file_path = os.path.join(output_directory, "interview_analysis_results.json")
        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {output_file_path}")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()