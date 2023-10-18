import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv2.imread('RISDance.jpg', 0)
m,n = img.shape
fig = plt.figure()
ax = plt.axes(projection='3d')
ker = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]])

ker_sizes = list(range(3, 16))
scales = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
X, Y  = np.meshgrid(np.log2(scales), ker_sizes)
Z = []
for r in ker_sizes:
    Z.append([])
    for s in scales:
        sum = 0
        for i in range(5):
            img_resize = cv2.resize(src=img, dsize=(0,0), fx = s, fy = s,
                                    interpolation=cv2.INTER_NEAREST)
            start_time = time.time()
            cv2.filter2D(kernel=ker, ddepth=-1, src=img_resize,
                         borderType=cv2.BORDER_DEFAULT)
            sum += time.time()-start_time
        Z[-1].append(sum/5)

ax.plot_surface(X, Y, np.array(Z), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.show()



