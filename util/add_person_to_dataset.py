import cv2
import os
import time
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
clear = lambda: os.system('cls')
from classes.camera import Camera
from src.misc.save_pre_processed_images import PreProcessAndSaveImagesNewPerson

def getLatestID(path):
    return len(os.listdir(path))

def main(dataset_path, person_first_name, person_last_name, names_txt_path,already_in_set=False, id=None):
    '''
    Opens the camera and lets you take picutres while its running, press space to take a picture, esc to exit the programm.
    @param dataset_path: The path to the dataset directory
    @param person_first_name: The first name of the new person. Leave blank if already_in_set = False!
    @param person_last_name: The last name of the new person. Leave blank if already_in_set = False!
    @param names_txt_path: The path to the names.txt file
    @param already_in_set: If you want to add more pictures to an already existing person
    @param id: Only use if you add more pictures to an already existing person! Add the id of the person from the dataset
    '''
    cam = Camera([640,480])
    dataset_path_image = dataset_path + "/images"
    dataset_path_pp = dataset_path + "/pre_processed"
    if not already_in_set:
        new_id = getLatestID(dataset_path_image) + 1
        img_counter = 1
        try:
            with open(names_txt_path, "a") as f:
                f.write(f"{person_first_name} {person_last_name} {new_id}\n")
            os.mkdir(os.path.join(dataset_path_image, str(new_id)))
            os.mkdir(os.path.join(dataset_path_pp, str(new_id)))
        except:
            print("[ERROR] Could not add person to name.txt!")
    elif id != None:
        img_counter = getLatestID(dataset_path_image + "/" + str(id)) + 1
        new_id = id
    else:
        print("Could not initialize! Exiting program!")
        return
    while True:
        start_time = time.time()
        #clear()
        ret, img = cam.cam.read()
        end_time = time.time()
        if ret:
            k = cv2.waitKey(1) 
            if k%256 == 27:
                # ESC PRESSED
                break
            elif k%256 == 32:
                # SPACE PRESSED
                saving_name = str(new_id)+"."+str(img_counter)+".jpg"
                saving_path = dataset_path_image + "/" + str(new_id) + "/"
                cam.takePicture(img, saving_name, saving_path)
                # Debug saving path of image
                #print(saving_path + saving_name)
                img_counter += 1
            try:
                fps = 1 / (end_time - start_time)
            except:
                print("Can not calculate fps, due to division by 0!")
                
            print(f"[INFO] INPUTS: ESC = EXIT, SPACE = Take Picutre for {person_first_name} {person_last_name} ID:{new_id} - FPS: {fps} - Frame time: {end_time - start_time}")
            cv2.imshow('camera',img)
        print(f"{img_counter} images taken!")
    cam.cam.release()
    cv2.destroyAllWindows()
    PreProcessAndSaveImagesNewPerson(dataset_path_image, dataset_path_pp, new_id)
    print("\n[INFO] Exiting program")

#Test sample 
#main("C:/Users/Student/Documents/datasets/personal_set", "Max", "Schurle", "C:/Users/Student/Documents/datasets/personal_set/names.txt")