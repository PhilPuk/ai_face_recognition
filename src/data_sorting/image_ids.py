id_amount = 150 #10177 #max

def getIDS(path):
    # Collect all existing ids and their images
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
            print("Current state: " + str(i))
            for line in lines:
                #Split the file name from the id
                text = line.split()
                if text[1] == str(i):
                    # Add pictures to the list of its corresponding id
                    images[i].append(text[0])
    return images, ids