import time
import datetime

def msgWithTime(message, code_index):
    '''
    @code 0 ERROR: catching and printing error
    @code 1 INFO: Info for programmer
    @code 2 USR_INFO: Info for program user
    '''
    codes = ["ERROR", "INFO", "USR_INFO"]
    curr_time = datetime.datetime.now()
    msg = f"{curr_time}: [{codes[code_index]}] {message}"
    print(msg)
    with open("./log.txt", 'a') as f:
        f.write(msg +"\n")
        
def logInput(message):
    curr_time = datetime.datetime.now()
    user_input = str(input(message))
    msg = f"{curr_time}: [USER_INPUT] {message} - INPUT: [{user_input}]"
    with open("./log.txt", 'a') as f:
        f.write(msg + "\n")
    return str(user_input)
    
    