from bs4 import BeautifulSoup

def extract_dataset_dir(xml_file):
    with open(xml_file) as f:
        # parse the XML file with BeautifulSoup
        soup = BeautifulSoup(f, 'xml')
        # find the dataset_dir attribute and get its value
        dataset_dir = soup.dataset['dataset_dir']
        # return the dataset_dir value
        return dataset_dir

dataset_dir = extract_dataset_dir('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml')
print(dataset_dir)
