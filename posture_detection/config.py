import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

VIDEO_SOURCE = 0  
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
