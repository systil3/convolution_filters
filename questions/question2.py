import cv2
import numpy as np

# Load the image
img = cv2.imread('vaultboy.png', 0)
ker = np.array([
    [1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0]])
#correlation
dst1 = cv2.filter2D(img, ddepth=-1,
                   kernel=ker,
                   borderType=cv2.BORDER_DEFAULT)
#convolution
dst2 = cv2.filter2D(img, ddepth=-1,
                   kernel=np.flip(np.flip(ker, axis=0), axis=1),
                   borderType=cv2.BORDER_DEFAULT)

cv2.imwrite('vaultboy_output1.png', dst1)
cv2.imwrite('vaultboy_output2.png', dst2)