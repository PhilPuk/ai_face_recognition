import new_data_loading_algo_2 as algo
import shutil

src_path = "C:/Users/Student/Documents/img_align_celeba/"
dst_path = "C:/Users/Student/Documents/sorted_dataset/"
images, ids = algo.getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")

for id in range(1, 10177, 1):
    dst_path = "C:/Users/Student/Documents/sorted_dataset/" + str(id)
    for image in images[id]:
        src_path = "C:/Users/Student/Documents/img_align_celeba/" + str(image)
        shutil.move(src_path,dst_path)
        
