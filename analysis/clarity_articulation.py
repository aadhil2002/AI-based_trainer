import numpy as np
import librosa
import logging
from typing import Dict
import json
import os

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


INPUT_DIR = r"D:\AI-based_trainer\data"
JSON_FILE = "interview_analysis_results.json"

def main():
    # Set up logging for debugging
    logging.basicConfig(level=logging.DEBUG)

    try:
        # Load the JSON file
        json_path = os.path.join(INPUT_DIR, JSON_FILE)
        logging.debug(f"Loading data from {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Iterate over each file's data in the JSON structure
        for file_name, modalities in data.items():
            logging.debug(f"Processing file: {file_name}")
            
            # Check if 'audio' data is present
            if 'audio' not in modalities:
                logging.debug(f"No audio data found for {file_name}")
                continue

            audio_data = modalities['audio']
            analyzer = ClarityArticulation()

            results = {}
            # Analyze each audio feature
            for feature_name, feature_values in audio_data.items():
                feature_array = np.array(feature_values)
                logging.debug(f"Analyzing feature: {feature_name}, shape: {feature_array.shape}")
                sr = 22050  # Assuming a common sample rate; adjust if different

                # Analyze the feature data
                result = analyzer.analyze(y=feature_array, sr=sr)
                results[feature_name] = result
                logging.debug(f"Analysis result for {feature_name}: {result}")

            # Optionally, save the results for each file
            output_path = os.path.join(INPUT_DIR, f"{file_name}_audio_analysis_results.json")
            with open(output_path, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            logging.debug(f"Results saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
