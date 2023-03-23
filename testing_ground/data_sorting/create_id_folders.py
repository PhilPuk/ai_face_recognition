#split dataset 70 training 30 testing
import os

new_directories = []
parent_dir = "C:/Users/Student/Documents/pre_processed_micro_data_set"

for i in range(1,6,1):
    new_directories.append(str(i))

for dir in new_directories:
    path = os.path.join(parent_dir, dir)
    os.mkdir(path)

