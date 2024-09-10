import os
import cv2
import json
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import scipy.signal
from typing import List, Dict, Tuple
import traceback
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for FFmpeg configuration
ffmpeg_configured = False

def configure_ffmpeg():
    global ffmpeg_configured
    if ffmpeg_configured:
        return

    def is_ffmpeg_available():
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # Try to use system PATH first
    ffmpeg_path = "ffmpeg"

    # If that doesn't work, specify the full path
    if not is_ffmpeg_available():
        ffmpeg_path = r"D:\tools\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"  # Adjust this path as needed
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")

    # Verify FFmpeg is available
    if not is_ffmpeg_available():
        raise EnvironmentError("FFmpeg is not available. Please check your installation.")

    logger.info("FFmpeg configuration completed successfully.")
    ffmpeg_configured = True

# Configuration dictionary
config = {
    'video_directory': r'D:\AI-based_trainer\dataset\Video_Song_Actor_01\Actor_01',
    'label_directory': r'D:\AI-based_trainer\dataset\Video_Song_Actor_01\Label',
    'file_extensions': ['.mp4', '.avi', '.mov'],
    'label_extension': '.json',  # Assuming labels are in JSON format
    'max_sample_files': 3,
    'error_log_file': 'video_reader_errors.log'
}

class VideoFileReader:
    def __init__(self, config: Dict):
        self.video_directory = config['video_directory']
        self.label_directory = config['label_directory']
        self.file_extensions = config['file_extensions']
        self.label_extension = config['label_extension']
        self.max_sample_files = config['max_sample_files']
        self.error_log_file = config['error_log_file']
        self.video_metadata: List[Dict] = []
        
        for directory in [self.video_directory, self.label_directory]:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory {directory} does not exist.")
            if not os.path.isdir(directory):
                raise NotADirectoryError(f"{directory} is not a directory.")
            
    def list_video_files(self) -> List[str]:
        """List all video files in the specified directory."""
        try:
            all_files = os.listdir(self.video_directory)
            video_files = [file for file in all_files 
                           if os.path.splitext(file)[1].lower() in self.file_extensions]
            if not video_files:
                raise FileNotFoundError("No video files found in the directory.")
            return video_files
        except PermissionError:
            self._log_error(f"Permission denied: Unable to access {self.video_directory}")
            return []
        except Exception as e:
            self._log_error(f"Error listing video files: {e}")
            return []

    def read_video_and_label_files(self) -> List[Dict]:
        """Read video files and their corresponding label files."""
        video_files = self.list_video_files()
        valid_data = []
        
        for video_file in video_files:
            video_path = os.path.join(self.video_directory, video_file)
            label_file = os.path.splitext(video_file)[0] + self.label_extension
            label_path = os.path.join(self.label_directory, label_file)
            
            try:
                # Read video metadata
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video file {video_path}")
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Read label file
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                
                # If both video and label are successfully read, add to valid data
                valid_data.append({
                    'video_file': video_file,
                    'video_path': video_path,
                    'label_file': label_file,
                    'label_path': label_path,
                    'label_data': label_data,
                    'duration': duration,
                    'frame_count': frame_count,
                    'fps': fps,
                    'resolution': f"{width}x{height}"
                })
                
            except FileNotFoundError:
                self._log_error(f"Label file not found for video: {video_file}")
            except json.JSONDecodeError:
                self._log_error(f"Error decoding JSON in label file: {label_file}")
            except cv2.error as e:
                self._log_error(f"OpenCV error reading video file {video_path}: {e}")
            except Exception as e:
                self._log_error(f"Error processing {video_file}: {e}")
        
        self.video_metadata = valid_data
        return valid_data

    def summary(self) -> None:
        """Provide a summary report of the loaded videos and labels."""
        if not self.video_metadata:
            logger.info("No valid video and label pairs loaded.")
            return
        
        total_pairs = len(self.video_metadata)
        total_duration = sum(video['duration'] for video in self.video_metadata)
        avg_duration = total_duration / total_pairs
        
        summary_report = (
            f"Summary Report:\n"
            f"Total valid video and label pairs: {total_pairs}\n"
            f"Total video duration: {total_duration:.2f} seconds\n"
            f"Average video duration: {avg_duration:.2f} seconds\n"
            f"\nSample Pairs:\n"
        )
        
        for i, data in enumerate(self.video_metadata[:self.max_sample_files], 1):
            summary_report += (
                f"{i}. Video: {data['video_file']}\n"
                f"   Label: {data['label_file']}\n"
                f"   Duration: {data['duration']:.2f} seconds\n"
                f"   Frame Count: {data['frame_count']}\n"
                f"   FPS: {data['fps']:.2f}\n"
                f"   Resolution: {data['resolution']}\n"
                f"   Label Data Sample: {str(data['label_data'])[:100]}...\n"
            )
        
        logger.info(summary_report)

    def _log_error(self, error_message: str) -> None:
        """Log error messages to the specified error log file."""
        with open(self.error_log_file, 'a') as f:
            f.write(f"{error_message}\n")
        logger.error(error_message)

class MultiModalProcessor(VideoFileReader):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.processed_data = []

    def process_files(self):
        """Process all valid video and label pairs."""
        valid_pairs = self.read_video_and_label_files()
        
        for data in valid_pairs:
            try:
                processed = self.process_single_file(data)
                if processed:
                    self.processed_data.append(processed)
            except Exception as e:
                logger.error(f"Error processing {data['video_file']}:")
                logger.error(traceback.format_exc())
                self._log_error(f"Error processing {data['video_file']}: {str(e)}")

    def process_single_file(self, data: Dict) -> Dict:
        """Process a single video file and its label."""
        file_name = data['video_file']
        video_path = data['video_path']
        
        logger.info(f"Processing file: {file_name}")
        logger.info(f"Full path: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Error: File does not exist: {video_path}")
            return None
        
        # Process visual data
        visual_data, visual_label = self.process_visual(video_path)
        
        # Process audio data
        audio_data, audio_label = self.process_audio(video_path)
        
        # Process text data (assuming speech-to-text)
        text_data, text_label = self.process_text(video_path)
        
        # Get overall label from the label file
        overall_label = data['label_data'].get('overall_label', None)
        
        return {
            'file_name': file_name,
            'data': (visual_data, visual_label, audio_data, audio_label, text_data, text_label, overall_label)
        }

    def process_visual(self, video_path: str) -> Tuple[np.ndarray, List[str]]:
        """Process visual data from the video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (224, 224)))  # Resize for consistency
        cap.release()
        
        visual_data = np.array(frames)
        # Placeholder for visual labels - you might want to implement actual labeling logic
        visual_label = ['visual_placeholder'] * len(frames)
        
        return visual_data, visual_label

    def process_audio(self, video_path: str) -> Tuple[np.ndarray, List[str]]:
        """Process audio data from the video."""
        try:
            # Load audio using pydub
            audio = AudioSegment.from_file(video_path, format="mp4")
        
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
        
            # If stereo, convert to mono
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
        
            # Normalize
            samples = samples / np.max(np.abs(samples))
        
            # Compute spectrogram
            _, _, spec = scipy.signal.spectrogram(samples, fs=audio.frame_rate)
        
            # Convert to dB scale
            spec_db = 10 * np.log10(spec + 1e-9)
        
            # Placeholder for audio labels
            audio_label = ['audio_placeholder'] * spec_db.shape[1]
        
            return spec_db, audio_label
        except Exception as e:
            logger.error(f"Error processing audio in {video_path}: {type(e).__name__}: {str(e)}")
            return np.array([]), []

    def process_text(self, video_path: str) -> Tuple[str, List[str]]:
        """Process text data (speech-to-text) from the video."""
        r = sr.Recognizer()
        
        try:
            # Extract audio from video
            audio = AudioSegment.from_file(video_path, format="mp4")
            audio_file = 'temp_audio.wav'
            audio.export(audio_file, format="wav")
            
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
            
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            text = ""
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)  # Clean up temporary file
        
        # Placeholder for text labels - you might want to implement actual labeling logic
        text_label = ['text_placeholder'] * len(text.split())
        
        return text, text_label

    def get_processed_data(self) -> List[Dict]:
        """Return the processed data."""
        return self.processed_data

    def summary(self) -> None:
        """Provide a summary report of the processed data."""
        if not self.processed_data:
            logger.info("No processed data available.")
            return
        
        total_files = len(self.processed_data)
        
        summary_report = (
            f"Summary Report:\n"
            f"Total processed files: {total_files}\n"
            f"\nSample Processed Data:\n"
        )
        
        for i, data in enumerate(self.processed_data[:self.max_sample_files], 1):
            file_name = data['file_name']
            visual_data, visual_label, audio_data, audio_label, text_data, text_label, overall_label = data['data']
            
            summary_report += (
                f"{i}. File: {file_name}\n"
                f"   Visual data shape: {visual_data.shape}\n"
                f"   Audio data shape: {audio_data.shape}\n"
                f"   Text data length: {len(text_data)}\n"
                f"   Overall label: {overall_label}\n"
            )
        
        logger.info(summary_report)

# Example usage
if __name__ == "__main__":
    try:
        configure_ffmpeg()
        processor = MultiModalProcessor(config)
        logger.info("Initialized MultiModalProcessor")
        
        logger.info("Processing files...")
        processor.process_files()
        
        logger.info("Generating summary...")
        processor.summary()
        
        processed_data = processor.get_processed_data()
        logger.info(f"Total processed data points: {len(processed_data)}")
        
        if len(processed_data) == 0:
            logger.warning("No data was processed successfully. Check the error log for details.")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(traceback.format_exc())