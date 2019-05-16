import cv2
from skimage.measure import compare_ssim


def compare_image(sourceImg, targetImg):

    imageA = cv2.imread(sourceImg, 0)
    imageB = cv2.imread(targetImg, 0)

    (score, diff) = compare_ssim(imageA, imageB, full=True)
    print("SSIM: {}".format(score))
    return score