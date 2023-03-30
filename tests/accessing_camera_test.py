import cv2

# Name of our camera window
cv2.namedWindow("preview")
# Start camera
vc = cv2.VideoCapture(0)
# Camera main loop
while True:
    # Read frame by frame from camera
    ret, frame = vc.read()
    # If frame exists
    if ret:
        # Show image in our window
        cv2.imshow("preview", frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
# Free memory and close programm
vc.release()
cv2.destroyWindow("preview")