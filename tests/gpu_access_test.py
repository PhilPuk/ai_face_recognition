import cv2
print(cv2.__version__)
from cv2 import cuda

#Should print very detailed report about the gpu if cmake build is correct
cuda.printCudaDeviceInfo(0)