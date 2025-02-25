import config

def analyze_posture(body_landmarks, hand_landmarks):
    """
    Analyze the posture and return feedback based on detected landmarks.
    """
    feedback = "Good posture detected"

    # Define key points
    left_shoulder = [body_landmarks[config.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     body_landmarks[config.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    right_shoulder = [body_landmarks[config.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      body_landmarks[config.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    left_wrist = [body_landmarks[config.mp_pose.PoseLandmark.LEFT_WRIST].x,
                  body_landmarks[config.mp_pose.PoseLandmark.LEFT_WRIST].y]
    right_wrist = [body_landmarks[config.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   body_landmarks[config.mp_pose.PoseLandmark.RIGHT_WRIST].y]
    left_hip = [body_landmarks[config.mp_pose.PoseLandmark.LEFT_HIP].x,
                body_landmarks[config.mp_pose.PoseLandmark.LEFT_HIP].y]
    right_hip = [body_landmarks[config.mp_pose.PoseLandmark.RIGHT_HIP].x,
                 body_landmarks[config.mp_pose.PoseLandmark.RIGHT_HIP].y]
    left_ear = body_landmarks[config.mp_pose.PoseLandmark.LEFT_EAR].y
    right_ear = body_landmarks[config.mp_pose.PoseLandmark.RIGHT_EAR].y

    detected_postures = []

    # Crossed arms detection
    wrist_distance = abs(left_wrist[0] - right_wrist[0])
    if wrist_distance < 0.05 and left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
        detected_postures.append("Arms crossed detected")

    # Hands on hips detection
    left_hand_on_hip = abs(left_wrist[1] - left_hip[1]) < 0.05
    right_hand_on_hip = abs(right_wrist[1] - right_hip[1]) < 0.05
    if left_hand_on_hip and right_hand_on_hip:
        detected_postures.append("Hands on waist detected")

    # Hair touching detection
    left_hand_near_head = abs(left_wrist[1] - left_ear) < 0.05
    right_hand_near_head = abs(right_wrist[1] - right_ear) < 0.05
    if left_hand_near_head or right_hand_near_head:
        detected_postures.append("Hair touching detected")

    # Open hands detection (only if hands are raised above hips)
    if hand_landmarks:
        for hand in hand_landmarks:
            if len(hand) >= 5:  # Checking if enough fingers are detected
                if left_wrist[1] < left_hip[1] or right_wrist[1] < right_hip[1]:  # Hands above hips
                    detected_postures.append("Open hands detected")

    # If no specific posture is detected, return default "Good posture detected"
    if detected_postures:
        feedback = ", ".join(detected_postures)

    return feedback
