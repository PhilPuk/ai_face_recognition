import sys
import os
sys.path.append("./")
from util.utils import msgWithTime
def createNewDataSetDir():
    os.mkdir(os.path.join("", "my_dataset"))
    os.mkdir(os.path.join("my_dataset/","images"))
    os.mkdir(os.path.join("my_dataset/","pre_processed"))
    msgWithTime("Create my_dataset, my_dataset/images and my_dataset/pre_processed folders.", 1)
    with open("my_dataset/names.txt", "w") as f:
        msgWithTime("Created names.txt file.", 1)

#createNewDataSetDir()