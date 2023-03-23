'''from sortedcontainers import SortedDict
map = SortedDict()
img_list = ['a','b']
img_list2 = ['d','z','e']
map['1'] = img_list
map['2'] = img_list2

print(map['1'],map['2'])
'''

id_amount = 10177 # max

def getIDS(path):
    # Collect all existing ids
    images = [[]]
    ids = []
    for i in range(id_amount+1):
        ids.append(i)
        img_l = []
        images.append(img_l)
    # Open txt with automatic closer
    with open(path) as f:
        # Collect all lines of the txt
        lines = f.readlines()
        for i in range(id_amount+1):
            print("Current state: " + str(round(float(i/(id_amount+1)*100),3)) + "%")
            for line in lines:
                #Split the file name from the id
                text = line.split()
                if text[1] == str(i):
                    # Add pictures to the list of its corresponding id
                    images[i].append(text[0])
                    lines.remove(line)
    return images, ids

#Test sample
images, ids = getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")
for i in range(id_amount+1):
    print("ID: " + str(i) + " - pictures:")
    print(images[i])
    print("\n\n")