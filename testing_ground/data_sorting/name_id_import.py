import os
path ="C:/Users/Student/Documents/datasets/personal_set/"
with open(path + "names.txt","r") as file:
    lines = file.readlines()
    print(lines[0].split())
