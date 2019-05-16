import cv2
import numpy as np
from matplotlib import pyplot as plt

def getOrigin():
    imgeWid  =600
    imgeHgt =600
    imge = np.zeros((imgeHgt,imgeWid), dtype=np.int);

    for c in range(1,10):
        for b in range(1,10):
            indy = int(c*imgeHgt/10)
            indx = int(b*imgeWid/10)
            for l in range(-2,3):
                for g in range(-2, 3):
                    imge[indy+l][indx+g] = 255

    cv2.imwrite("originFile.png", imge)

def GaussMix(filepath):
    imge = cv2.imread(filepath)

    kernel2 = np.array(
        [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273.0
    gaussian = cv2.filter2D(imge, -1, kernel2)

    cv2.imwrite(filepath, gaussian)

def FuncGaussMix(filepath, func, funcSum):
    imge = cv2.imread(filepath)

    kernel2 = np.array(
        [[func[-2][2], func[-1][2], func[0][2], func[1][2], func[2][2]],
         [func[-2][1], func[-1][1], func[0][1], func[1][1], func[2][1]],
         [func[-2][0], func[-1][0], func[0][0], func[1][0], func[2][0]],
         [func[-2][-1], func[-1][-1], func[0][-1], func[1][-1], func[2][-1]],
         [func[-2][-2], func[-1][-2], func[0][-2], func[1][-2], func[2][-2]]]) / funcSum
    funcGaussian = cv2.filter2D(imge, -1, kernel2)

    cv2.imwrite(filepath, funcGaussian)

def getAffine():
    pts1 = np.float32([[50, 50], [50, 550], [550, 50], [550, 550]])
    pts2 = np.float32([[100, 100], [50, 300], [300, 50], [550, 550]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return M



if __name__ == '__main__':
    # GaussMix("originFile.png")
    img = cv2.imread("originFile.png")
    rows, cols, ch = img.shape
    dst = cv2.warpPerspective(img, getAffine(), (cols, rows))
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    plt.show()



