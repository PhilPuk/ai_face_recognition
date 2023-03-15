from bs4 import BeautifulSoup

def extract_dataset_dir(xml_file="C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml", data="training"):
    '''
    Returns value of the path to the datasets.
    Takes an xml file path.
    Takes "training" or "testing" or "training_pre_processed" as string for choosing between datasets
    '''
    with open(xml_file) as f:
        # parse the XML file with BeautifulSoup
        soup = BeautifulSoup(f, 'xml')
        # find the dataset_dir attribute and get its value
        dataset_dir = soup.dataset[f'{data}_data_dir']
        # return the dataset_dir value
        return dataset_dir