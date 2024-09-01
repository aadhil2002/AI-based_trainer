import sys
import traceback
import logging
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, List
import cv2
import librosa
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoFeatureExtractor, AutoModelForAudioClassification,
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSequenceClassification
)

# Set environment variable for PyTorch
os.environ["USE_TORCH"] = "1"

from analysis.engagement_interaction import EngagementInteraction
from analysis.confidence_level import ConfidenceLevel
from analysis.voice_modulation import VoiceModulation
from analysis.fluency_intonation import FluencyIntonation
from analysis.clarity_articulation import ClarityArticulation
from analysis.emotional_expression import EmotionalExpression
from analysis.phrasing_language import PhrasingLanguage
from preprocessing.FeatureExtractor import MultimodalFeaturePipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for input and output directories
INPUT_DIR = r"D:\trainer\data"
OUTPUT_DIR = r"D:\trainer\data"

def log_exception(e):
    logging.error(f"An error occurred: {str(e)}")
    logging.error("Traceback:")
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    for line in tb_lines:
        logging.error(line.rstrip())

class InferencePipeline:
    def __init__(self):
        logging.debug("Initializing InferencePipeline")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)
        self.image_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
        
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)
        
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
        
        self.text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.text_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base").to(self.device)

        self.engagement_interaction = EngagementInteraction()
        self.confidence_level = ConfidenceLevel()
        self.voice_modulation = VoiceModulation()
        self.fluency_intonation = FluencyIntonation()
        self.clarity_articulation = ClarityArticulation()
        self.emotional_expression = EmotionalExpression()
        self.phrasing_language = PhrasingLanguage()
        self.feature_extractor = MultimodalFeaturePipeline()
        logging.debug("InferencePipeline initialized")

    def process_interview(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug("Starting process_interview")
        results = {}
        total_files = len(preprocessed_data)
        logging.info(f"Total files to process: {total_files}")

        with tqdm(total=total_files, desc="Processing interviews") as pbar:
            for filename, data in preprocessed_data.items():
                logging.info(f"Processing file: {filename}")
                if len(data) != 7:
                    logging.error(f"Incorrect data format for file {filename}. Expected 7 elements, got {len(data)}")
                    continue
                visual_data, visual_label, audio_data, audio_label, text_data, text_label, overall_label = data

                logging.debug(f"Data shapes - Visual: {np.array(visual_data).shape}, Audio: {np.array(audio_data).shape}, Text: {len(text_data)}")

                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_data = {
                        executor.submit(self.process_video, visual_data): "video",
                        executor.submit(self.process_audio, audio_data): "audio",
                        executor.submit(self.process_text, text_data): "text"
                    }

                    video_results, audio_results, text_results = {}, {}, {}

                    for future in as_completed(future_to_data, timeout=300):  # 5-minute timeout
                        data_type = future_to_data[future]
                        try:
                            if data_type == "video":
                                video_results = future.result()
                            elif data_type == "audio":
                                audio_results = future.result()
                            else:
                                text_results = future.result()
                            logging.debug(f"Completed processing {data_type} for {filename}")
                        except TimeoutError:
                            logging.error(f"Processing {data_type} data for {filename} timed out")
                        except Exception as e:
                            logging.error(f"Error processing {data_type} data for {filename}: {str(e)}")
                            logging.exception(e)

                results[filename] = self.combine_results(video_results, audio_results, text_results, {
                    "visual": visual_label,
                    "audio": audio_label,
                    "text": text_label,
                    "overall": overall_label
                })
                pbar.update(1)
                logging.info(f"Completed processing file: {filename}")

        logging.debug("Finished process_interview")
        return results

    def process_video(self, visual_data: List[List[List[float]]]) -> Dict[str, Any]:
        logging.debug("Starting process_video")
        try:
            results = {
                "engagement": [],
                "confidence": [],
                "facial_expressions": [],
                "visual_features": []
            }

            for i, frame in enumerate(tqdm(visual_data, desc="Processing visual", leave=False)):
                logging.debug(f"Processing frame {i+1}/{len(visual_data)}")
                frame_uint8 = np.clip(np.array(frame) * 255, 0, 255).astype(np.uint8)
                if frame_uint8.shape[-1] == 3:  # If RGB
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                else:  # If grayscale
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

                results["engagement"].append(self.engagement_interaction.analyze(frame_bgr))
                results["confidence"].append(self.confidence_level.analyze(frame_bgr))
                results["facial_expressions"].append(self.confidence_level.analyze_facial_expressions(frame_bgr))
                
                # Use ViT model for visual feature extraction
                inputs = self.image_processor(images=frame_uint8, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.image_model(**inputs)
                visual_features = outputs.logits.cpu().numpy()
                results["visual_features"].append(visual_features.tolist())

            logging.debug("Finished process_video")
            return results
        except Exception as e:
            logging.error(f"Error in video processing: {str(e)}")
            logging.exception(e)
            return {}

    def process_audio(self, audio_data: List[List[float]]) -> Dict[str, Any]:
        logging.debug("Starting process_audio")
        try:
            results = {
                "voice_modulation": [],
                "fluency": [],
                "clarity": [],
                "emotional_expression": [],
                "audio_features": [],
                "transcription": []
            }

            audio_np = np.array(audio_data)
            if audio_np.ndim == 2:
                audio_np = np.mean(audio_np, axis=0)  # Convert stereo to mono

            # Ensure minimum audio length of 0.5 seconds
            min_duration = 0.5
            min_samples = int(16000 * min_duration)
            if len(audio_np) < min_samples:
                logging.warning(f"Audio too short. Padding to {min_duration} seconds.")
                audio_np = np.pad(audio_np, (0, min_samples - len(audio_np)))

            # Process the entire audio as one chunk if it's shorter than 5 seconds
            chunk_duration = max(5, len(audio_np) / 16000)
            chunk_samples = int(chunk_duration * 16000)
            num_chunks = 1 if len(audio_np) <= chunk_samples else int(np.ceil(len(audio_np) / chunk_samples))

            logging.debug(f"Audio shape: {audio_np.shape}, Number of chunks: {num_chunks}")

            for i in tqdm(range(num_chunks), desc="Processing audio", leave=False):
                start = i * chunk_samples
                end = min((i + 1) * chunk_samples, len(audio_np))
                chunk = audio_np[start:end]

                logging.debug(f"Processing chunk {i+1}/{num_chunks}, shape: {chunk.shape}")

                # Adjust n_fft based on chunk length
                n_fft = min(2048, len(chunk))
                logging.debug(f"Using n_fft: {n_fft}")

                results["voice_modulation"].append(self.voice_modulation.analyze(chunk, 16000, n_fft=n_fft))
                results["fluency"].append(self.fluency_intonation.analyze(chunk, 16000, n_fft=n_fft))
                results["clarity"].append(self.clarity_articulation.analyze(chunk, 16000, n_fft=n_fft))
                results["emotional_expression"].append(self.emotional_expression.analyze_audio(chunk, 16000))

                # Use AST model for audio feature extraction
                inputs = self.audio_feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.audio_model(**inputs)
                audio_features = outputs.logits.cpu().numpy()
                results["audio_features"].append(audio_features.tolist())

                # Use Whisper model for transcription
                input_features = self.whisper_processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                with torch.no_grad():
                    generated_ids = self.whisper_model.generate(input_features)
                transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results["transcription"].append(transcription)

            logging.debug("Finished process_audio")
            return results
        except Exception as e:
            logging.error(f"Error in audio processing: {str(e)}")
            logging.exception(e)
            return {}

    def process_text(self, text_data: List[str]) -> Dict[str, Any]:
        logging.debug("Starting process_text")
        try:
            results = {
                "phrasing_language": self.phrasing_language.analyze(" ".join(text_data)),
                "emotional_expression": [],
                "text_features": []
            }

            for sentence in tqdm(text_data, desc="Processing text", leave=False):
                # Emotion analysis using the emotion-english-distilroberta-base model
                inputs = self.text_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                emotion_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                emotion_label = self.text_model.config.id2label[emotion_scores.argmax().item()]
                results["emotional_expression"].append({"emotion": emotion_label, "scores": emotion_scores.cpu().numpy().tolist()[0]})

                # Text feature extraction
                text_features = outputs.logits.cpu().numpy()
                results["text_features"].append(text_features.tolist())

            logging.debug(f"Processed {len(text_data)} text sentences")
            logging.debug("Finished process_text")
            return results
        except Exception as e:
            logging.error(f"Error in text processing: {str(e)}")
            logging.exception(e)
            return {}

    def combine_results(self, video_results: Dict[str, Any], audio_results: Dict[str, Any], 
                        text_results: Dict[str, Any], labels: Dict[str, str]) -> Dict[str, Any]:
        logging.debug("Combining results")
        combined = {
            "engagement_and_interaction": {
                "video": video_results.get("engagement", []),
            },
            "confidence_level": {
                "video": video_results.get("confidence", []),
                "audio": audio_results.get("voice_modulation", [])
            },
            "voice_modulation": audio_results.get("voice_modulation", []),
            "fluency_and_intonation": audio_results.get("fluency", []),
            "clarity_and_articulation": audio_results.get("clarity", []),
            "emotional_expression": {
                "video": video_results.get("facial_expressions", []),
                "audio": audio_results.get("emotional_expression", []),
                "text": text_results.get("emotional_expression", [])
            },
            "phrasing_and_language": text_results.get("phrasing_language", {}),
            "features": {
                "visual": video_results.get("visual_features", []),
                "audio": audio_results.get("audio_features", []),
                "text": text_results.get("text_features", [])
            },
            "transcription": audio_results.get("transcription", []),
            "labels": labels
        }

        logging.debug("Finished combining results")
        return combined

def main():
    logging.debug("Entering main function")
    try:
        start_time = time.time()
        pipeline = InferencePipeline()
        logging.info("Pipeline initialized")

        logging.info("Loading preprocessed data")
        preprocessed_data_path = os.path.join(INPUT_DIR, "preprocessed_data.pkl")
        if not os.path.exists(preprocessed_data_path):
            logging.error(f"File not found: {preprocessed_data_path}")
            return

        try:
            with open(preprocessed_data_path, "rb") as f:
                preprocessed_data = pickle.load(f)
            logging.info(f"Preprocessed data loaded. Total items: {len(preprocessed_data)}")
            if not isinstance(preprocessed_data, dict):
                logging.error(f"Unexpected data type in {preprocessed_data_path}. Expected dict, got {type(preprocessed_data)}")
                return
        except Exception as e:
            logging.error(f"Error loading preprocessed data: {str(e)}")
            log_exception(e)
            return

        logging.info("Starting interview processing")
        results = pipeline.process_interview(preprocessed_data)

        logging.info("Analysis complete. Saving results.")
        try:
            output_path = os.path.join(OUTPUT_DIR, "interview_analysis_results.json")
            os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            log_exception(e)

        end_time = time.time()
        logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logging.error("Error in main function")
        log_exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Unhandled exception in script")
        log_exception(e)
    finally:
        logging.debug("Reached end of script")
        logging.shutdown()