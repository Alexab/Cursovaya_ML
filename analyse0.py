import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1_clr = cv.imread('images/w_4.png')
img1_clr = cv.cvtColor(img1_clr,cv.COLOR_BGR2HSV)
img2_clr = cv.imread('images/w_9.png')
img2_clr = cv.cvtColor(img2_clr,cv.COLOR_BGR2HSV)


hist1 = cv.calcHist([img1_clr],[0],None,[10],[0,179])
hist2 = cv.calcHist([img2_clr],[0],None,[10],[0,179])

hist3 = cv.calcHist([img1_clr],[1],None,[10],[0,256])
hist4 = cv.calcHist([img2_clr],[1],None,[10],[0,256])

hist5 = cv.calcHist([img1_clr],[2],None,[10],[0,256])
hist6 = cv.calcHist([img2_clr],[2],None,[10],[0,256])


plt.subplot(231), plt.plot(hist1), plt.title('H'), plt.ylabel('количество пикселей')
plt.subplot(232), plt.plot(hist2), plt.title('S')
plt.subplot(233), plt.plot(hist3), plt.title('V')
plt.subplot(234), plt.plot(hist4), plt.ylabel('количество пикселей')
plt.subplot(235), plt.plot(hist5)
plt.subplot(236), plt.plot(hist6)

plt.show()



