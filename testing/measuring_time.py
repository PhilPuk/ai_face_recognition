import time
import os

clear = lambda: os.system('cls')
register = [0,0,0]

for i in range(1,100000000,1):
    start = time.time() 
    clear()
    average_time = sum(register) / len(register)
    print(f"{i}: average_time: {round(average_time * 1000,2)}")
    end = time.time()
    register.insert(0,end-start)
    if len(register) > 20:
        del register[-1]
