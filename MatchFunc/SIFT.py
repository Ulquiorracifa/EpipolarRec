import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def SIFTMatch(sourceImg, targetImg):
    img1 = cv2.imread(sourceImg, 0)  # queryImage
    img2 = cv2.imread(targetImg, 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    hmerge = np.hstack((img3, img4))
    cv2.imshow("point", hmerge)  # 拼接显示为gray
    cv2.waitKey(0)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imshow("BFmatch", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()