import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from typing import List, Tuple, Dict

class PostureTracker:
    """A class tracking and analyzing body posture and gestures using MediaPipe."""
    
    def __init__(self, model_path: str = 'body_language_new.pkl'):
        """
        Initializing the PostureTracker with MediaPipe components and ML model.
        
        Args:
            model_path: Path to the pickled body language classification model
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        
        # Initialize MediaPipe components
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.5)
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        # Load ML model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Initialize tracking variables
        self.left_wrist_frames: List = []
        self.right_wrist_frames: List = []
        self.frames_since_gesture: int = 0
        
    def analyze_vector_change(self, frames: List[List[float]], threshold: float = 0.15) -> List[int]:
        """
        Analyzing the change in vectors between consecutive frames.
        
        Args:
            frames: List of position vectors
            threshold: Minimum angle change to be considered significant
            
        Returns:
            List of indices where significant changes occurred
        """
        angle_changes = []
        
        for i in range(1, len(frames)):
            v1, v2 = frames[i - 1], frames[i]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle_changes.append(angle)
            
        return [i for i, angle in enumerate(angle_changes) if angle > threshold]
    
    def is_gesture(self, frames: List[List[float]], threshold: float = 0.15) -> bool:
        """
        Detect if a gesture occurred in the most recent frames.
        
        Args:
            frames: List of position vectors
            threshold: Minimum angle change to be considered a gesture
            
        Returns:
            Boolean indicating if a gesture was detected
        """
        if len(frames) < 2:
            return False
            
        v1 = frames[-2]
        v2 = frames[-1]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        return angle > threshold
    
    def is_facing_forward(self, left_z: float, right_z: float) -> bool:
        """
        Determine if the person is facing forward based on shoulder positions.
        
        Args:
            left_z: Z-coordinate of left shoulder
            right_z: Z-coordinate of right shoulder
            
        Returns:
            Boolean indicating if person is facing forward
        """
        if (left_z < 0 and right_z > 0) or (left_z > 0 and right_z < 0):
            return False
            
        if (left_z > 0 and right_z > 0) or (left_z < 0 and right_z < 0):
            return abs(left_z - right_z) <= 0.37
            
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame and detect posture/gestures.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of processed frame and detection results
        """
        # Convert color space for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detections
        pose_results = self.pose.process(image)
        face_results = self.holistic.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not pose_results.pose_landmarks:
            return image, {}
            
        # Process landmarks
        landmarks = pose_results.pose_landmarks.landmark
        
        # Track wrist movements
        left_wrist = [round(landmarks[15].x, 2), round(landmarks[15].y, 2), round(landmarks[15].z, 2)]
        right_wrist = [round(landmarks[16].x, 2), round(landmarks[16].y, 2), round(landmarks[16].z, 2)]
        
        self.left_wrist_frames.append(left_wrist)
        self.right_wrist_frames.append(right_wrist)
        
        # Detect gestures
        if len(self.right_wrist_frames) > 2 and len(self.left_wrist_frames) > 2:
            if self.is_gesture(self.right_wrist_frames) or self.is_gesture(self.left_wrist_frames):
                self.left_wrist_frames = []
                self.right_wrist_frames = []
                self.frames_since_gesture = 0
            else:
                self.frames_since_gesture += 1
        
        # Check posture
        if self.is_facing_forward(landmarks[12].z, landmarks[11].z):
            shoulder_z = (landmarks[11].z + landmarks[12].z) / 2
            waist_z = (landmarks[23].z + landmarks[24].z) / 2
            
            if abs(shoulder_z) / abs(waist_z) > 400:
                self._draw_warning(image, "STAND UP STRAIGHT!", landmarks[11])
        
        # Process face and pose for ML model
        if face_results.pose_landmarks and face_results.face_landmarks:
            pose_row = self._extract_pose_data(face_results.pose_landmarks.landmark)
            face_row = self._extract_face_data(face_results.face_landmarks.landmark)
            
            # Make prediction
            X = pd.DataFrame([pose_row + face_row])
            prediction = self.model.predict(X.values)[0]
            probability = self.model.predict_proba(X.values)[0]
            
            if probability[np.argmax(probability)] > 0.7:
                self._draw_prediction(image, prediction, probability)
        
        # Draw pose landmarks
        self._draw_pose_landmarks(image, pose_results)
        
        return image, {
            'frames_since_gesture': self.frames_since_gesture,
            'facing_forward': self.is_facing_forward(landmarks[12].z, landmarks[11].z)
        }
    
    def _extract_pose_data(self, landmarks) -> List[float]:
        """Extract pose landmark data."""
        return list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                            for landmark in landmarks]).flatten())
    
    def _extract_face_data(self, landmarks) -> List[float]:
        """Extract face landmark data."""
        return list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                            for landmark in landmarks]).flatten())
    
    def _draw_warning(self, image: np.ndarray, text: str, landmark) -> None:
        """Draw warning text on the image."""
        position = tuple(np.multiply([landmark.x, landmark.y], [640, 480]).astype(int))
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 250), 2, cv2.LINE_AA)
    
    def _draw_prediction(self, image: np.ndarray, prediction: str, probability: np.ndarray) -> None:
        """Draw ML model prediction on the image."""
        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
        
        # Draw class
        cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, prediction.split(' ')[0], (90, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw probability
        cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(probability[np.argmax(probability)], 2)), 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    def _draw_pose_landmarks(self, image: np.ndarray, results) -> None:
        """Draw pose landmarks on the image."""
        self.mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

def main():
    """Main function to run the posture tracking system."""
    tracker = PostureTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, results = tracker.process_frame(frame)
            
            if results.get('frames_since_gesture', 0) > 40:
                cv2.putText(processed_frame, "MOVE YOUR HANDS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA)
            
            cv2.imshow('Posture Tracker', processed_frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
