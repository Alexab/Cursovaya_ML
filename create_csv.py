import pandas as pd
import numpy as np
import cv2 as cv


weather = 1


df = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0]], columns=['par1', 'par2', 'par3', 'par4', 'par5',
                                                    'par6','par7','par8','par9','weather'])

img1_clr = cv.imread('carla3.png')
img1_clr = cv.cvtColor(img1_clr, cv.COLOR_BGR2HSV)
hist_1 = cv.calcHist([img1_clr],[0],None,[10],[0,179])
hist_1_av = cv.calcHist([img1_clr],[0],None,[179],[0,179])
hist_2 = cv.calcHist([img1_clr],[1],None,[10],[0,256])
hist_2_av = cv.calcHist([img1_clr],[1],None,[256],[0,256])
hist_3 = cv.calcHist([img1_clr],[2],None,[10],[0,256])
hist_3_av = cv.calcHist([img1_clr],[2],None,[256],[0,256])

sum = 0
for num in range(len(hist_1_av)):
    sum += hist_1_av[num][0]*num
par1 = int(sum/480000) # Среднее значение по всему изображению
par2 = np.where(hist_1 == max(hist_1))[0][0] # Номер диапазона с максиамльным значение мпикселей
par3 = hist_1[5][0]

sum = 0
for num in range(len(hist_2_av)):
    sum += hist_2_av[num][0]*num
par4 = int(sum/480000) # Среднее значение по всему изображению
par5 = np.where(hist_2 == max(hist_2))[0][0] # Номер диапазона с максиамльным значение мпикселей
par6 = hist_2[5][0]

sum = 0
for num in range(len(hist_3_av)):
    sum += hist_3_av[num][0]*num
par7 = int(sum/480000) # Среднее значение по всему изображению
par8 = np.where(hist_3 == max(hist_3))[0][0] # Номер диапазона с максиамльным значение мпикселей
par9 = hist_3[5][0]



df2 = pd.DataFrame([[par1, par2, par3, par4, par5, par6, par7, par8, par9, weather]],
                   columns = ['par1', 'par2', 'par3', 'par4', 'par5', 'par6','par7','par8','par9','weather'])
df = df.append(df2, ignore_index=True)

#Не забыть удалить первую сьолку
print(df.head())