import cv2

class Camera():
    def __init__(self, resolution):
        self.width = resolution[0]
        self.height = resolution[1]
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3,self.width)
        self.cam.set(4,self.height)
        
    def takePicture(self, img, saving_name, saving_path):
        '''
        Saves a pictures at the given path with the given name:
        @param saving_name: Name of image to save, should include name and image extension.
        @param saving_path: Path where to save the picutre, end path with / !
        '''
        cv2.imwrite(saving_path+saving_name, img)