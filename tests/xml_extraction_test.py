import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from data_sorting.extract_dataset_directory import extract_dataset_dir

dataset_dir = extract_dataset_dir('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml')
print(dataset_dir)
