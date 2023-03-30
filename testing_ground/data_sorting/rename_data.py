import os

def renameFilesInDirectory(path, new_name_base, file_extension):
    '''
    Renames files in given directory with the given base name such as: base.1, base.2, base.3...
    Parameters: path: Path of directory, new_name_base: replaces base in example
    '''
    count = 1
    for file in os.listdir(path):
        new_name = path + new_name_base + "." + str(count) + file_extension
        source = path + file
        print("Old file name: ",source)
        print("New file name: ",new_name)
        print("\n")
        os.rename(source, new_name)
        count+=1
        
# Test sample         
renameFilesInDirectory("./my_dataset/pre_processed/1/", "1", ".jpeg")