from config import mp_pose

def analyze_posture(landmarks):
    """Analyzes different postures based on detected landmarks."""
    
    # Extract key points
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_EYE].y]
    right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y]

    feedback = "Good posture ✅"

    # Crossed arms detection
    wrist_distance = abs(left_wrist[0] - right_wrist[0])
    if wrist_distance < 0.05:
        feedback = "Arms crossed detected ✅"

    # Hands on hips detection
    left_hand_on_hip = abs(left_wrist[1] - left_hip[1]) < 0.05
    right_hand_on_hip = abs(right_wrist[1] - right_hip[1]) < 0.05
    if left_hand_on_hip and right_hand_on_hip:
        feedback = "Hands on hips detected ✅"

    # Touching hair detection
    left_hand_near_head = abs(left_wrist[1] - left_eye[1]) < 0.05
    right_hand_near_head = abs(right_wrist[1] - right_eye[1]) < 0.05
    if left_hand_near_head or right_hand_near_head:
        feedback = "Warning: Touching hair 🚨"

    return feedback
