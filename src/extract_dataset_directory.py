from bs4 import BeautifulSoup

#Returns value of the path to the datasets.
#Takes an xml file path.
#Takes "training" or "testing" as string for choosing between datasets
def extract_dataset_dir(xml_file, training_or_testing):
    with open(xml_file) as f:
        # parse the XML file with BeautifulSoup
        soup = BeautifulSoup(f, 'xml')
        # find the dataset_dir attribute and get its value
        dataset_dir = soup.dataset[f'{training_or_testing}_data_dir']
        # return the dataset_dir value
        return dataset_dir