import numpy as np
from PIL import Image
import os

'''
def getIDS(path):
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            print(line)
'''     
id_amount = 100 #10177
images = [[]]
ids = []
'''
def getIDS(path):
    # Collect all existing ids
    for i in range(id_amount):
        ids.append(i)
    # Open txt with automatic closer
    with open(path) as f:
        # Collect all lines of the txt
        lines = f.readlines()
        for i in range(id_amount):
            print("Current state: " + str(i))
            img_list = []
            for line in lines:
                #Split the file name from the id
                text = line.split()
                if text[1] == str(i):
                    # Add pictures to the list of its corresponding id
                    img_list.append(text[0])
            images.append(img_list)
'''
def getIDS(path):
    # Collect all existing ids
    for i in range(id_amount+1):
        ids.append(i)
        img_l = []
        images.append(img_l)
    # Open txt with automatic closer
    with open(path) as f:
        # Collect all lines of the txt
        lines = f.readlines()
        for i in range(id_amount+1):
            print("Current state: " + str(i))
            for line in lines:
                #Split the file name from the id
                text = line.split()
                if text[1] == str(i):
                    # Add pictures to the list of its corresponding id
                    images[i].append(text[0])
            
getIDS("C:/Users/Student/Documents/AI_Face_Recognition/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")

for i in range(id_amount+1):
    print("ID: " + str(i) + " - pictures:")
    print(images[i])
    print("\n\n")
        