import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()
    # If frame exists
    if ret:
        cv2.imshow("preview", frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
# Free memory and close programm
vc.release()
cv2.destroyWindow("preview")