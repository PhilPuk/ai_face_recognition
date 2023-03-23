import cv2
print(cv2.__version__)

app_name = 'Face_Recognition'
cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cv2.namedWindow(app_name)
vc = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = vc.read()
    if not ret:
        print("Empty frame")
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Algorithm to detect faces
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display resulting frame
        cv2.imshow(app_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free memory and close programm
vc.release()
cv2.destroyAllWindows()