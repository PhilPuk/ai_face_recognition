import cv2

app_name = 'preview'
cascadePath = "./dataset/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cv2.namedWindow(app_name)
vc = cv2.VideoCapture(0)
# Camera loop
while True:
    # Capture frame-by-frame from camera
    ret, frame = vc.read()
    # Check if camera frame exists
    if ret:
        # Grayscale image from camera
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
    # Press ESC to end
    key = cv2.waitKey(20)
    if key == 27:
        break

# Free memory and close programm
vc.release()
cv2.destroyAllWindows()