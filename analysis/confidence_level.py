import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Any

class ConfidenceLevel:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    def analyze(self, frame: np.ndarray) -> Dict[str, float]:
        try:
            face_results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not face_results.multi_face_landmarks or not pose_results.pose_landmarks:
                return {"confidence_score": 0.5}

            facial_confidence = self.analyze_facial_expressions(frame)['confidence']

            shoulder_left = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            posture_score = 1 - abs(shoulder_left.y - shoulder_right.y)

            confidence_score = (facial_confidence + posture_score) / 2
            return {"confidence_score": float(confidence_score)}
        except Exception as e:
            logging.error(f"Error in ConfidenceLevel analysis: {str(e)}")
            return {"confidence_score": 0.5}

    def analyze_facial_expressions(self, frame: np.ndarray) -> Dict[str, Any]:
        face_results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not face_results.multi_face_landmarks:
            return {"expression": "unknown", "confidence": 0.5}

        landmarks = face_results.multi_face_landmarks[0].landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        eye_distance = ((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)**0.5
        mouth_width = ((left_mouth.x - right_mouth.x)**2 + (left_mouth.y - right_mouth.y)**2)**0.5

        if mouth_width > 0.5:
            expression = "smile"
        elif eye_distance < 0.2:
            expression = "focused"
        else:
            expression = "neutral"

        confidence = min(mouth_width + eye_distance, 1.0)

        return {"expression": expression, "confidence": confidence}