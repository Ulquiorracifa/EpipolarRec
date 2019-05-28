import cv2
import numpy as np
import math

pointsPair = np.array([[50, 50], [50, 550], [550, 50], [550, 550]])/600
repointPair = np.array([[200, 50], [50, 550], [400, 50], [550, 550]])/600

originR = 2 #核半径
defineR_zero = 3**2
sigma = 1




def maconFromFunc(image, Func, funcNum):

    image = np.array(image)
    w, h =image

    MR = PerspfunPointsPair(0, True)  # 透视逆变换
    M = PerspfunPointsPair(0, False)
    for c in range(h):
        for b in range(w):
            imageSim = np.matrix(MR)* np.matrix([c,b]).T #模拟图对应点
            Round = max(((np.matrix(M) * np.matrix(imageSim + [originR, originR]).T -np.matrix([c,b])).fabs()).getA()+
                        ((np.matrix(M) * np.matrix(imageSim + [-originR, originR]).T-np.matrix([c,b])).fabs()).getA()+
                        ((np.matrix(M) * np.matrix(imageSim + [originR, -originR]).T-np.matrix([c,b])).fabs()).getA()+
                        ((np.matrix(M) * np.matrix(imageSim + [-originR, -originR]).T-np.matrix([c,b])).fabs()).getA())
            Round = math.ceil(Round)
            guassSum = 0
            for x in range(0,2*Round+1):
                for y in range(0,2*Round+1):
                    tpSim = np.matrix(MR)* np.matrix([c+x, y]).T
                    dltaDis = (tpSim - np.matrix([c+x,y])).getA()
                    dltaDis = sum([c*c for c in dltaDis])
                    if dltaDis > defineR_zero:
                        guassWeight = 0
                    else:
                        guassWeight = np.exp(-0.5*dltaDis/(sigma**2))
                    guassSum += guassWeight
                    # jacobianWeight =






    return


def fun(c):
    funMatrix = np.zeros((5,5))

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)


    return funMatrix

def PerspfunPointsPair(testNum, size, reverse=False):
    #size 2
    if testNum == 0:
        pspOrigin = pointsPair[0]*size
        pspSimulate = repointPair[0]*size

    if reverse==True:
        tmp = pspOrigin
        pspOrigin = pspSimulate
        pspSimulate = tmp

    return cv2.getPerspectiveTransform(pspOrigin, pspSimulate)

if __name__ =='__main__':
    maconFromFunc()