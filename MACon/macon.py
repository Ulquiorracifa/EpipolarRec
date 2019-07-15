import cv2
import numpy as np
import math
import sympy

pointsPair = np.float32([[50, 50], [50, 550], [550, 50], [550, 550]]) / 600
repointsPair = np.float32([[200, 50], [50, 550], [400, 50], [550, 550]]) / 600
# pointsPair = np.float32([[50, 50], [50, 550], [550, 50], [550, 550]]) / 600
# repointsPair = np.float32([[50, 50], [50, 550], [550, 50], [550, 550]]) / 600
originR = 2  # 核半径(5*5)
defineR_zero = 3 ** 2
sigma = 1


def getJacobianFromPairs(x, y):
    # x2 = (x1-300)*(3/2500*y1+17/50)+300    ---->f
    # y2 = y1                                ---->g
    # dy1/dx1 = 0
    # return h[[dy1/dx1,dy1/dx2],[dy2/dx1,dy2/dx2]]

    Jh = (3 / 2500 * y + 17 / 50)

    return Jh


def maconFromFunc(image):
    image = np.array(image)
    w, h = image.shape

    # MR = PerspfunPointsPair(0, 600, reverse=True)  # 透视逆变换
    # M = PerspfunPointsPair(0, 600, reverse=False)
    # print(MR)
    # print(M)
    aimImage = np.zeros([h, w], np.float64)
    for c in range(h):
        for b in range(w):
            # imageSim = cv2.perspectiveTransform(np.float32([[c,b]]), np.array(repointsPair*600),np.array(pointsPair*600))
            # imageSim = np.matrix(MR) * np.matrix([b, c, 1]).T  # 模拟图对应点
            # imageSim = imageSim.reshape([1, 3])
            # print(imageSim)
            # print((np.matrix(M) * np.matrix(imageSim + [originR, originR, 0]).T).reshape([1, 3]))
            # Round = max(
            #     np.fabs(
            #         np.hstack(
            #             ((np.matrix(M) * np.matrix(imageSim + [originR, originR, 0]).T - np.matrix([c, b, 1])).getA(),
            #              (np.matrix(M) * np.matrix(imageSim + [-originR, originR, 0]).T - np.matrix([c, b, 1])).getA(),
            #              (np.matrix(M) * np.matrix(imageSim + [originR, -originR, 0]).T - np.matrix([c, b, 1])).getA(),
            #              (np.matrix(M) * np.matrix(imageSim + [-originR, -originR, 0]).T - np.matrix(
            #                  [c, b, 1])).getA()))))
            imageSim = getPer((b, c), True)  # 模拟图对应点
            if imageSim[0]<0 or imageSim[0]>w or imageSim[1]<0 or imageSim[1]>h:
                continue

            p1 = getPer(imageSim + [originR, originR], False)-[b,c]
            p2 = getPer(imageSim + [originR, -originR], False)-[b,c]
            p3 = getPer(imageSim + [-originR, originR], False)-[b,c]
            p4 = getPer(imageSim + [-originR, -originR], False)-[b,c]
            Round = max(
                np.fabs(
                    np.hstack(
                        (p1, p2, p3, p4))))
            Round = math.ceil(Round)
            guassSum = 0
            pixelSum = 0
            for y in range(0, 2 * Round + 1):
                for x in range(0, 2 * Round + 1):
                    y_ = y - Round
                    x_ = x - Round

                    yi = c + y_
                    xj = b + x_
                    if xj < 0 or xj >= w or yi < 0 or yi >= h:
                        continue

                    # tpSim = np.matrix(MR) * np.matrix([c + x, y]).T
                    tpSim = getPer([b+x, c+y], True)
                    dltaDis = tpSim - imageSim
                    dltaDis = sum([c * c for c in dltaDis])
                    if dltaDis > defineR_zero:
                        guassWeight = 0
                    else:
                        guassWeight = np.exp(-0.5 * dltaDis / (sigma ** 2))
                    guassSum += guassWeight
                    jacobianWeight = getJacobianFromPairs(xj, yi)
                    # jacobianWeight =1
                    pixelSum += guassWeight * jacobianWeight * image[yi][xj]
            aimImage[c][b] = pixelSum / guassSum

    return aimImage


# def fun(c):
#     funMatrix = np.zeros((5,5))
#
#     pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
#     pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
#
#     M = cv2.getAffineTransform(pts1, pts2)
#
#
#     return funMatrix

def PerspfunPointsPair(testNum, size, reverse=False):
    # size 2
    if testNum == 0:
        pspOrigin = pointsPair * size
        pspSimulate = repointsPair * size

    if reverse == True:
        tmp = pspOrigin
        pspOrigin = pspSimulate
        pspSimulate = tmp
    # print(pspOrigin)
    # print(pspSimulate)

    return cv2.getPerspectiveTransform(pspOrigin, pspSimulate)


def getPer(point, rev=False):
    x1, y1 = point[:2]
    if rev == False:
        x2 = (x1 - 300) * (3 / 2500 * y1 + 17 / 50) + 300
        y2 = y1
    else:
        x2 = (x1 - 300) / (3 / 2500 * y1 + 17 / 50) + 300
        y2 = y1

    return np.array([x2, y2])
    # return np.array([x1, y1])

def generPreGau():
    image = cv2.imread("D:\Development\pycharm_workspace\EpipolarRec\expData\originFile.png", cv2.IMREAD_GRAYSCALE)
    # res = maconFromFunc(image)
    # cv2.imwrite("D:\Development\pycharm_workspace\EpipolarRec\expData\MaDst.jpg", res)
    res = cv2.GaussianBlur(image, (3, 3), 1)
    rows, cols = res.shape
    dst = cv2.warpPerspective(res, cv2.getPerspectiveTransform(pointsPair * rows, repointsPair * rows), (cols, rows))
    cv2.imwrite("D:\Development\pycharm_workspace\EpipolarRec\expData\preGausDst.jpg", dst)

def macGau():
    image = cv2.imread("D:\Development\pycharm_workspace\EpipolarRec\expData\originFile.png", cv2.IMREAD_GRAYSCALE)
    res = maconFromFunc(image)
    cv2.imwrite("D:\Development\pycharm_workspace\EpipolarRec\expData\MaDst.jpg", res)

if __name__ == '__main__':
    macGau()
    # image = cv2.imread("D:\Development\pycharm_workspace\EpipolarRec\expData\originFile.png", cv2.IMREAD_GRAYSCALE)
    # res = maconFromFunc(image)
    # cv2.imwrite("D:\Development\pycharm_workspace\EpipolarRec\expData\MaDst.jpg", res)