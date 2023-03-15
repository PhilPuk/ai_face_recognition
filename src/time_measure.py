import time
from cv2 import cuda

class Measurer:
    def __init__(self, format="ms", register_precision=100):
        self.format = format
        self.register_precision = register_precision
        self.time_start = 0
        self.time_end = 0
        self.time_ms = 0
        self.time_s = 0
        self.time_average = 0
        self.time_register = [0,0,0]
        self.time_left = 0
        self.time_total = 0
        
    def measure_time_start(self):
        '''Use this function before any operation of the code you want to measure'''
        self.time_start = time.time()
        
    def measure_time_end(self):
        '''Use this function at the end of the operation of the code you want to measure'''
        self.time_end = time.time()
    
    def calculate_time(self, current_index, max_index):
        self.time_ms = self.time_end - self.time_start
        self.time_s = self.time_ms / 1000
        self.time_total += self.time_ms
        self.time_register.append(self.time_ms)
        if len(self.time_register) > self.register_precision:
            del self.time_register[-1]
        self.time_average = sum(self.time_register) / len(self.time_register)
        self.time_left = self.time_average * (max_index * current_index)
        
            