# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:31:14 2024

@author: User
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Загрузка изображения
    image = cv.cvtColor(cv.imread('winter_cat.png'), cv.COLOR_BGR2GRAY)
    plt.imshow(image, cmap = 'gray')
    plt.show()
    
    channels = [0]
    histSize = [256]
    qqq = [0, 256]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([256, 1]) 
    
    hsum = hist1.sum()
    for i in range(256):
        lut[i] = np.uint8(255 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    plt.imshow(image2, cmap = 'gray')
    plt.show()