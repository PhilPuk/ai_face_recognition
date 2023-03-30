import cv2 

def face_detection(img, gray_scaled_img,  putRectangle, minneig=5 , haardcascade_path="./dataset/haarcascade_frontalface_default.xml"):
    '''
    Returns passed image, faces tuple. Img with rectangles around faces, aswell as the vectors and coordinates for the detected faces such as: x,y,width,height.
    @param img: loaded img, with PIL, skimage or cv2, @param minneig: Higher numbers reduces chances of false detection.
    @param haarcascade_path: Path to file :"haarcascade_frontalface_default.xml". Downloadable in the internet, or included in cv2/data.
    '''
    face_cascade = cv2.CascadeClassifier(haardcascade_path)

    # Detect faces  
    faces = face_cascade.detectMultiScale(
            gray_scaled_img,     
            scaleFactor=1.2,
            minNeighbors=minneig,     
            minSize=(20, 20)
        ) 
    if putRectangle:
        # Draw rectangle around the faces  
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    
    return img, faces

def main(cam, haarcascade_path):
    cam.cam = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, img = cam.cam.read()
        if not ret:
            print("Empty frame")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #Algorithm to detect faces
            img, faces = face_detection(img, gray,True, 6, haardcascade_path=haarcascade_path)
            # Draw a rectangle around the faces
            # Display resulting frame
            cv2.imshow("camera", img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    # Free memory and close programm
    cam.cam.release()
    cv2.destroyAllWindows()
