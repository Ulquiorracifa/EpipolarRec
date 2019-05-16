import cv2
import numpy as np

def maconFromFunc(image, Func, ReFunc):



    return

def fun(c):
    funMatrix = np.zeros((5,5))

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)


    return funMatrix