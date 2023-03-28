import cv2
from skimage import io 
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from src.facial_recognition import face_recognition_image as rec
from src.facial_recognition.recognizer.recognizer_2 import Model
print("Model loading!")
model = Model(0, True)
print("Model loaded!")
def predict_and_show_img(img_path, model):
    '''
    Predicts who the person(s) on the image is(are) and show the image in a window.
    @param img_path: The path of the image to predict of.
    @param model: The model class 
    '''
    img = io.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img, faces = rec.face_detecting(img, gray_img, minneig=7)
    # Display the output
    img = rec.face_recognition_image(img,gray_img, faces, model.recognizer)
    rec.printingWindow(img)

predict_and_show_img("C:/Users/Student/Desktop/edited_images/0.4.png", model)

# Latest point: Model error. create new model. probably can delete old model. # next model --> try more data and less training.