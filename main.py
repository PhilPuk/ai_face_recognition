import sys
import traceback
from skimage import io
import cv2
sys.path.append('./')
from classes.model import Model
from classes.camera import Camera
from util.utils import msgWithTime, logInput, printingWindow
from Detection import face_detection
from Image_Recognition import face_recognition
from Camera_Recognition import cam_recognition
from util import add_person_to_dataset
from src.misc import access_camera, get_person_dict

'''
Driver program, combining all functions of this project.
'''

dataset_path = "./my_dataset"
names_txt_path = "./my_dataset/names.txt"
camera_resolution = [640,480]
haarCascadePath = "./dataset/haarcascade_frontalface_default.xml"
model_path = "./models/my_model.xml"
predict_img_path = "./images/phil_p_3.jpg"
trained_epochs = 0
try:
    loaded_img = io.imread(predict_img_path)
except:
    msgWithTime("Could not load testing image!",0)
try:
    person_dict = get_person_dict.getNamesAndIDSfromTXT(names_txt_path)
    msgWithTime(f"Loaded person dict: {person_dict}", 1)
except:
    msgWithTime("Could not load person_dict!", 0)

def Dataset(cam, model):
    global dataset_path
    global haarCascadePath
    def addPersonToDataset():
        first_name = logInput("Input First Name:")
        last_name = logInput("Input: Last Name:")
        add_person_to_dataset.main(dataset_path=dataset_path,
                                   person_first_name=first_name,
                                   person_last_name=last_name,
                                   names_txt_path=names_txt_path,
                                   cam=cam,
                                   already_in_set=False,
        )
    while True:
        # Output for user
        msgWithTime("Dataset Settings", 2)
        msgWithTime(f"Current Datasetpath: {dataset_path}", 2)
        msgWithTime(f"Current HaarCascadePath: {haarCascadePath}", 2)
        msgWithTime("(1) Add new person to dataset, (2) Change Datasetpath, (3) Change HaarCascade Path (4) Go back", 2)
        
        usr_input = logInput("Input: ")
        if usr_input == "1":
            addPersonToDataset()
            model.createModel(model.model_path, dataset_path)
        elif usr_input == "2":
            dataset_path = logInput("Input new datasetpath: ")
        elif usr_input == "3":
            haarCascadePath = logInput("Input new datasetpath: ")
        elif usr_input == "4":
            return
        else:
            msgWithTime("Invalid user input", 0)
        
def Modell(model):
    global trained_epochs
    global haarCascadePath
    global dataset_path
    global model_path
    msgWithTime(f"Current model path: {model_path}",2)
    msgWithTime("(1) Change model path and load model, (2) Train Model, (3) Go back", 2)
    usr_input = logInput("Input: ")
    if usr_input == "1":
        model_path = logInput("Input new model path: ")
        try:
            model.loadModel(model_path)
        except:
            msgWithTime("Could not load model from given path.", 0)
    elif usr_input == "2":
        epoch_amount = logInput("Input amount of epochs: ")
        if epoch_amount.isnumeric() == False:
            epoch_amount = 3
        else:
            epoch_amount = int(epoch_amount)
            trained_epochs += epoch_amount
            msgWithTime("Could not change epoch amount, now training for 3 epochs!", 0)
        msgWithTime("Do not stop this process until the training is finished! Otherwise the model could become corrupted!", 2)
        model.trainExistingModel(dataset_path, epoch_amount, trained_epochs)
        msgWithTime("Reloading trained model.",2)
        model.loadModel(model_path)
    elif usr_input == "3":
        return
    else:
        msgWithTime("Invalid user input", 0)

def Detection(cam_object):
    global haarCascadePath
    msgWithTime("Initializing Detection!", 2)
    face_detection.main(cam_object, haarCascadePath)

def Recognition(model, cam):
    global names_txt_path
    global haarCascadePath
    global predict_img_path
    global loaded_img
    global person_dict
    #Recognition for images and webcam
    msgWithTime("Recognition", 2)
    msgWithTime("(1) Predict on webcam, (2) Predict on Images, (3) Set Image path to predict, (4) Show current image, (5) Go back", 2)
    usr_input = logInput("Input: ")
    if usr_input == "1":
        try:
            msgWithTime("Initializing Predicting on webcam!", 2)
            cam_recognition.main(model, cam, person_dict)
        except Exception:
            msgWithTime(f"{traceback.format_exc()}", 0)
            msgWithTime("Could not run webcam prediction due to error above!", 0)
    elif usr_input == "2":
        try:
            tmp_img = loaded_img.copy()
            gray_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)
            predicted_img = face_recognition.face_recognition_image(loaded_img, gray_img,model.recognizer,person_dict)
            printingWindow(predicted_img, window_title="Predicted Image")
            loaded_img = tmp_img.copy()
        except Exception:
            msgWithTime()
    elif usr_input == "3":
        try:
            predict_img_path = logInput("Input new image path: ")
            loaded_img = io.imread(predict_img_path)
        except:
            msgWithTime(f"{traceback.format_exc()}", 0)
            msgWithTime(f"Could not run prediction on image, due to error above", 0) 
    elif usr_input == "4":
        printingWindow(loaded_img, window_title="Current selected Image")
    elif usr_input == "5":
        return
    else:
        msgWithTime("Invalid user input", 0)

def ChangeCamera(cam_object):
    global camera_resolution
    msgWithTime("Camera Settings", 2)
    msgWithTime(f"Current Camera resolution: {camera_resolution}, Recommended: [640,480]", 2)
    msgWithTime("(1) Change Camera resolution, (2) Test camera, (3) Go back", 1)
    usr_input = logInput("Input: ")
    if usr_input == "1":
        camera_resolution[0] = logInput("Set camera x: ")
        camera_resolution[1] = logInput("Set camera y: ")  
        cam_object.cam.set(3,int(camera_resolution[0])) #width
        cam_object.cam.set(4,int(camera_resolution[1])) #height      
    elif usr_input == "2":
        access_camera.main(cam_object)
    elif usr_input == "3":
        return
    else:
        msgWithTime("Invalid user input", 0)

def Help():
    msgWithTime("No help yet:", 2)

def main():
    global model_path
    global haarCascadePath
    global names_txt_path
    try:
        msgWithTime("Initializing program!", 1)
        # Everything to initialize before main loop
        model = Model(model_path, haarCascadePath, names_txt_path)
        cam = Camera([640,480])
        msgWithTime("Finished initializing! Programm starting!", 1)
        msgWithTime("Welcome to AI_FACE_Recognition!", 1)
        while True:
            msgWithTime("(1) Dataset (2) Modell, (3) Detection, (4) Recognition, (5) Camera, (6) Help, (7) Exit", 2)
            input = logInput("Input: ")
            if input == "1":
                Dataset(cam, model)
            elif input == "2":
                Modell(model)
            elif input == "3":
                Detection(cam)
            elif input == "4":
                Recognition(model, cam)
            elif input == "5":
                ChangeCamera(cam)
            elif input == "6":
                Help()
            elif input == "7":
                break
            else:
                msgWithTime("Invalid input, pls try again!", 0)
        msgWithTime("Exiting programm.", 1)
    except Exception:
        msgWithTime(f"{traceback.format_exc()}", 0)
        msgWithTime(f"Exited programm by error above", 0)  
main()