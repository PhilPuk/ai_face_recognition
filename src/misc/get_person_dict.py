def getNamesAndIDSfromTXT(names_txt_path):
    with open(names_txt_path, "r") as f:
        lines = f.readlines()
        persons = dict()
        for line in lines:
            info = line.split()
            persons[info[2]] = f"{info[0]} {info[1]}"
        return persons