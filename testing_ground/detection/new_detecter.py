import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions_drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()
        image= cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landsmarks:
            for face_landmarks in results.multi_face_landsmakrs:
                #print(face_landmarks)
                mp_drawing.draw_landmarks(image,landmark_list=face_landmarks,connections=mp_face_mesh.FACE_CONNECTIONS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)
        end = time.time()
        totalTime = end - start
        print(f"Time of frame: {totalTime}")
        cap.imshow(image)
        