import cv2
import mediapipe as mp
import config
from posture_analysis import analyze_posture

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process pose detection
    results_pose = pose.process(image)
    
    # Process hand detection
    results_hands = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get landmarks if detected
    body_landmarks = results_pose.pose_landmarks.landmark if results_pose.pose_landmarks else None
    hand_landmarks = []

    if results_hands.multi_hand_landmarks:
        for hand in results_hands.multi_hand_landmarks:
            hand_landmarks.append(hand.landmark)

    # Analyze posture
    if body_landmarks:
        feedback = analyze_posture(body_landmarks, hand_landmarks)
        cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Changed color to red
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),  # Changed color to blue
                                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))  # Changed color to yellow
        
        # Draw hand landmarks if detected
        if results_hands.multi_hand_landmarks:
            for hand in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Changed color to green
                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))  # Changed color to cyan

    # Display the output
    cv2.imshow('Posture Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
