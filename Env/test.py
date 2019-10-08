import cv2
import numpy
cv2.namedWindow("i")
for i in range (10000):
    I=numpy.ones((1080,1000),dtype=numpy.uint8)*(i%255)
    cv2.imshow("i", I)
    cv2.waitKey(1)