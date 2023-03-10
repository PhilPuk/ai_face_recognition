'''
70% training data
30% testing data
move 30% of the pictures of each id into the testing data folder
'''

import new_data_loading_algo_2 as algo
import shutil
import os, os.path

images, ids = algo.getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")

for id in range(1, 10178, 1):
    src_path = "C:/Users/Student/Documents/training_data/" + str(id)
    dest_path = "C:/Users/Student/Documents/testing_data/" + str(id)
    #Count files in folder
    amount_of_Files = len(os.listdir(src_path))
    moving_files = int(amount_of_Files // 3)
    for i in range(0,moving_files,1):
        src_pic_path = "C:/Users/Student/Documents/training_data/" + str(id) + "/" + str(images[id][i])
        shutil.move(src_pic_path,dest_path)
    status = round(float(id/10178)*100,2)
    print("Status: " + str(status) + "%")