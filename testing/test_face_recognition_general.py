import cv2
print(cv2.__version__)

'''
latest debugging results:
    - cascadeclassifier maybe not correctly loaded?
    - find new xml file of haarcascade...
'''

app_name = 'Face_Recognition'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cv2.namedWindow(app_name)
vc = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = vc.read()
    if not ret:
        print("Empty frame")
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.rectangle(frame, (100, 200), (100+50, 200+50), (0, 255, 0), 2) #Testing rectangle
        # Display resulting frame
        cv2.imshow(app_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
vc.release()
cv2.destroyAllWindows()