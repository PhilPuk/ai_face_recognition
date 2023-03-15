#split dataset 70 training 30 testing
import os

new_directories = []
parent_dir = "C:/Users/Student/Documents/training_data_pre_processed"

for i in range(1,10178,1):
    new_directories.append(str(i))

for dir in new_directories:
    path = os.path.join(parent_dir, dir)
    os.mkdir(path)

