import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict

class EngagementInteraction:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    def analyze(self, frame: np.ndarray) -> Dict[str, float]:
        try:
            pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not pose_results.pose_landmarks:
                return {"engagement_score": 0.5}

            left_eye = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE]
            right_eye = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE]
            shoulder_mid = (pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y +
                            pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            eye_level = (left_eye.y + right_eye.y) / 2
            eye_shoulder_ratio = abs(eye_level - shoulder_mid) / max(abs(left_eye.x - right_eye.x), 1e-5)

            engagement_score = 1 - min(eye_shoulder_ratio, 1)
            return {"engagement_score": float(engagement_score)}
        except Exception as e:
            logging.error(f"Error in EngagementInteraction analysis: {str(e)}")
            return {"engagement_score": 0.5}