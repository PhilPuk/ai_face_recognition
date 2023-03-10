#Return images in a list of list, with the ids corresponding to the lists
def getIDS(path):
    # Collect all existing ids
    images = [[]]
    ids = []
    id_amount = 0
    with open(path) as f:
        for line in f:
            # Split the file name from the id
            text = line.split()
            # Update the number of ids if necessary
            if int(text[1]) > id_amount:
                id_amount = int(text[1])
                # Add empty lists for the new ids
                for i in range(len(ids) + 1, id_amount + 2):
                    ids.append(i - 1)
                    images.append([])
            # Add pictures to the list of its corresponding id
            images[int(text[1])].append(text[0])
    return images, ids


#Test sample
images, ids = getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")
images_total = 0
for i in range(len(ids)):
    print("ID: " + str(i) + " - pictures:")
    print(images[i])
    print("\n\n")