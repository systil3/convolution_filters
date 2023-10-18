import cv2
import numpy as np

# Load the image
img = cv2.imread('diy.jpg', 0)
img2 = cv2.imread('aphext.jpg', 0).astype(np.uint8)
ker_hp = np.array([
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]])
ker_lp = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]])

dst1_sobel = cv2.Sobel(img, ddepth=-1,
                    dx = 1, dy = 1,
                   borderType=cv2.BORDER_DEFAULT)
dst1_canny = cv2.Canny(img, threshold1=100, threshold2=255)
dst1_lap = cv2.Laplacian(img, ksize=3, ddepth=-1, borderType=0)
dst2_blur = cv2.blur(img2, ksize=(5, 5), borderType=0)
dst2_gauss = cv2.GaussianBlur(img2, sigmaX=10, sigmaY=10, ksize=(5, 5), borderType=0)

cv2.imwrite('diy_output_sobel.png', dst1_sobel)
cv2.imwrite('diy_output_canny.png', dst1_canny)
cv2.imwrite('diy_output_laplacian.png', dst1_lap)
cv2.imwrite('aphext.jpg', img2)
cv2.imwrite('aphext_output_blur.jpg', dst2_blur)
cv2.imwrite('aphext_output_gaussian.jpg', dst2_gauss)