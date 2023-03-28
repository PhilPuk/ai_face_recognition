import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from classes.model import Model
from classes.camera import Camera
from util.utils import msgWithTime, logInput

'''
Driver program, combining all functions of this project.
'''

def Dataset():
    msgWithTime("Dataset:", 2)

def Modell():
    msgWithTime("Modell:", 2)

def Detection():
    msgWithTime("Detection:", 2)

def Recognition():
    msgWithTime("Recognition:", 2)

def Help():
    msgWithTime("Help:", 2)

def main():
    msgWithTime("Initializing program!", 1)
    # Everything to initialize before main loop
    model = Model()
    cam = Camera([640,480])
    msgWithTime("Finished initializing! Programm starting!", 1)
    msgWithTime("Welcome to AI_FACE_Recognition!", 1)
    while True:
        msgWithTime("(1) Dataset (2) Modell, (3) Detection, (4) Recognition, (5) Help, (6) Exit", 2)
        input = logInput("Input: ")
        if input == "1":
            Dataset()
        elif input == "2":
            Modell()
        elif input == "3":
            Detection()
        elif input == "4":
            Recognition()
        elif input == "5":
            Help()
        elif input == "6":
            break
        else:
            msgWithTime("Invalid input, pls try again!", 0)
    msgWithTime("Exiting programm.", 1)
            
main()