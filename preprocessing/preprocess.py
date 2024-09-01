import numpy as np
from PIL import Image
import librosa
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import cv2
from scipy import signal
from GetData import MultiModalProcessor, config
import multiprocessing
import re
import pickle
import os
import logging
import traceback

class MultiModalPreprocessor:
    def __init__(self):
        self.preprocessed_data = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Download necessary NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

        # Initialize the MultiModalProcessor from GetData.py
        self.data_processor = MultiModalProcessor(config)

    def preprocess_all(self):
        # Process files using MultiModalProcessor
        self.data_processor.process_files()
        raw_data = self.data_processor.get_processed_data()

        logging.info(f"Total raw data points: {len(raw_data)}")

        # Use multiprocessing to parallelize preprocessing
        with multiprocessing.Pool() as pool:
            results = pool.map(self.preprocess_file, raw_data)

        # Collect results
        for result in results:
            if result is not None:
                file_name, processed_data = result
                self.preprocessed_data[file_name] = processed_data

    def preprocess_file(self, file_data):
        file_name = file_data['file_name']
        visual_data, visual_label, auditory_data, auditory_label, textual_data, textual_label, overall_label = file_data['data']

        try:
            processed_visual = self.preprocess_visual(visual_data)
            processed_audio = self.preprocess_audio(auditory_data)
            processed_text = self.preprocess_text(textual_data)

            # Extract label from filename
            extracted_label = self.extract_label_from_filename(file_name)

            # Process labels
            visual_label = self.process_label(visual_label, 'visual') or extracted_label
            auditory_label = self.process_label(auditory_label, 'audio') or extracted_label
            textual_label = self.process_label(textual_label, 'text') or extracted_label

            # Determine overall label
            if overall_label is None or 'placeholder' in str(overall_label).lower():
                overall_label = extracted_label
            else:
                overall_label = self.process_label(overall_label, 'overall')

            return file_name, (
                processed_visual, visual_label,
                processed_audio, auditory_label,
                processed_text, textual_label,
                overall_label
            )
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
            logging.error(f"Visual data shape: {visual_data.shape if isinstance(visual_data, np.ndarray) else 'N/A'}")
            logging.error(f"Audio data shape: {auditory_data.shape if isinstance(auditory_data, np.ndarray) else 'N/A'}")
            logging.error(f"Text data: {textual_data[:100] if isinstance(textual_data, str) else 'N/A'}")
            logging.error("Traceback: ", exc_info=True)
            return None

    def process_label(self, label, modality):
        if isinstance(label, list):
            non_placeholder = [item for item in label if 'placeholder' not in item.lower()]
            return non_placeholder[0] if non_placeholder else self.extract_label_from_filename(modality)
        return str(label) if label is not None else self.extract_label_from_filename(modality)

    def extract_label_from_filename(self, filename_or_modality):
        if isinstance(filename_or_modality, str) and '-' in filename_or_modality:
            # It's a filename
            parts = filename_or_modality.split('-')
            if len(parts) >= 6:
                emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
                emotion = emotion_map.get(parts[1], 'unknown')
                intensity = 'normal' if parts[2] == '01' else 'strong'
                return f"{emotion}_{intensity}"
        # It's a modality or unknown filename format
        return f"{filename_or_modality}_label"

    def preprocess_visual(self, visual_data):
        if visual_data is None or len(visual_data) == 0:
            raise ValueError("Visual data is missing or empty")

        processed_frames = []
        for frame in visual_data:
            image = Image.fromarray(frame)

            # Resize
            image = image.resize((224, 224))

            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0

            # Data augmentation
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)  # Horizontal flip
            if np.random.rand() > 0.5:
                img_array = np.rot90(img_array, k=1)  # 90-degree rotation

            processed_frames.append(img_array.tolist())  # Convert to list for JSON serialization

        return processed_frames

    def preprocess_audio(self, audio_data, sr=22050, target_sr=16000, duration=5):
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is missing or empty")

        # Ensure audio_data is 1D
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)

        # Resample to 16kHz
        audio_resampled = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)

        # Normalize
        audio_normalized = librosa.util.normalize(audio_resampled)

        # Apply high-pass filter
        nyquist = 0.5 * target_sr
        cutoff = 100 / nyquist
        b, a = signal.butter(5, cutoff, btype='high', analog=False)
        audio_filtered = signal.filtfilt(b, a, audio_normalized)

        # Ensure consistent length
        if len(audio_filtered) > target_sr * duration:
            audio_filtered = audio_filtered[:target_sr * duration]
        else:
            audio_filtered = librosa.util.fix_length(audio_filtered, size=target_sr * duration)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_filtered, sr=target_sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Data augmentation: Add small random noise
        noise = np.random.normal(0, 0.001, mel_spec_db.shape)
        mel_spec_db_augmented = mel_spec_db + noise

        return mel_spec_db_augmented.tolist()  # Convert to list for JSON serialization

    def preprocess_text(self, text_data):
        if text_data is None or not isinstance(text_data, str):
            raise ValueError("Invalid or missing text data")

        # Lowercase
        text_lower = text_data.lower()

        # Tokenize
        tokens = word_tokenize(text_lower)

        # Remove punctuation and stop words, but keep more informative words
        cleaned_tokens = [
            token for token in tokens
            if token not in string.punctuation and (len(token) > 2 or token in ['a', 'an', 'the'])
        ]

        # If the result is too short, return the original tokens without removing stop words
        if len(cleaned_tokens) < 3:
            return [token for token in tokens if token not in string.punctuation]

        return cleaned_tokens

    def validate_preprocessed_data(self):
        for file_name, data in self.preprocessed_data.items():
            visual_data, visual_label, audio_data, audio_label, text_data, text_label, overall_label = data

            if len(visual_data) == 0 or len(visual_data[0]) != 224 or len(visual_data[0][0]) != 224 or len(visual_data[0][0][0]) != 3:
                logging.warning(f"Warning: Unexpected visual data shape for {file_name}")

            if len(audio_data) != 128:
                logging.warning(f"Warning: Unexpected audio data shape for {file_name}")

            if len(text_data) < 2:
                logging.warning(f"Warning: Very short text data for {file_name}")

            if 'placeholder' in overall_label.lower() or 'unknown' in overall_label.lower():
                logging.warning(f"Warning: Potentially incorrect overall label for {file_name}: {overall_label}")

    def get_preprocessed_data(self):
        return self.preprocessed_data

    def save_preprocessed_data(self, directory='D:\AI-based_trainer\data', filename='preprocessed_data.pkl'):
        """
        Save the preprocessed data to a file using pickle.

        Args:
        directory (str): The directory to save the file in.
        filename (str): The name of the file to save the data to.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Join the directory and filename
            full_path = os.path.join(directory, filename)
            
            with open(full_path, 'wb') as f:
                pickle.dump(self.preprocessed_data, f)
            logging.info(f"Preprocessed data saved successfully to {full_path}")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {str(e)}")

    def load_preprocessed_data(self, directory='D:\AI-based_trainer\data', filename='preprocessed_data.pkl'):
        """
        Load preprocessed data from a file.

        Args:
        directory (str): The directory to load the file from.
        filename (str): The name of the file to load the data from.

        Returns:
        bool: True if data was loaded successfully, False otherwise.
        """
        full_path = os.path.join(directory, filename)
        
        if not os.path.exists(full_path):
            logging.warning(f"File {full_path} does not exist.")
            return False

        try:
            with open(full_path, 'rb') as f:
                self.preprocessed_data = pickle.load(f)
            logging.info(f"Preprocessed data loaded successfully from {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading preprocessed data: {str(e)}")
            return False

if __name__ == "__main__":
    preprocessor = MultiModalPreprocessor()
    
    # Specify the directory to save/load the preprocessed data
    data_directory = r'D:\AI-based_trainer\data'

    # Check if preprocessed data file exists
    if not preprocessor.load_preprocessed_data(directory=data_directory):
        # If not, preprocess the data
        preprocessor.preprocess_all()
        preprocessor.validate_preprocessed_data()
        # Save the preprocessed data to the specified directory
        preprocessor.save_preprocessed_data(directory=data_directory)

    preprocessed_data = preprocessor.get_preprocessed_data()

    logging.info(f"Total preprocessed data points: {len(preprocessed_data)}")

    # Print a sample of preprocessed data
    if preprocessed_data:
        sample_key = next(iter(preprocessed_data))
        sample_data = preprocessed_data[sample_key]
        logging.info(f"\nSample preprocessed data for {sample_key}:")
        logging.info(f"Visual data shape: {np.array(sample_data[0]).shape}")
        logging.info(f"Visual label: {sample_data[1]}")
        logging.info(f"Audio data shape: {np.array(sample_data[2]).shape}")
        logging.info(f"Audio label: {sample_data[3]}")
        logging.info(f"Text data (first 10 tokens): {sample_data[4][:10]}")
        logging.info(f"Text label: {sample_data[5]}")
        logging.info(f"Overall label: {sample_data[6]}")
    else:
        logging.warning("No data was preprocessed successfully. Check for errors in processing.")