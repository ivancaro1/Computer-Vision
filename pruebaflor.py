# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:46:26 2021

@author: ivan
"""

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

shape=cv.MORPH_RECT#  MORPH_RECT  MORPH_ELLIPSE  MORPH_CROSS
ksize=(5,5)
kernel=cv.getStructuringElement(shape,ksize)#ancla en centro por defecto.


imagen = 'livel_cell.jpg'
imag = cv.imread(imagen,1)

if imag is None:
    print("error en imagen")
    exit()
  
    
imb, img, imr = cv.split(imag)

closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

plt.imshow(closing)
plt.show()

def Umbral_super_lento_pero_facil_de_entender(T1, T2, image):
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            # threshold
            image[y, x] = 255 if (image[y, x] >= T1 and image[y,x] <= T2) else 0
    return image

umbral = Umbral_super_lento_pero_facil_de_entender(200, 255, closing)

plt.imshow(umbral)
plt.show()






erosion = cv.erode(imag,kernel,iterations = 1)

dilation = cv.dilate(umbral,kernel,iterations = 1)




#cv2.imshow("erosion",erosion)

#cv2.imshow("Original",imag)

cv.imshow("dilation",dilation)

#cv2.imshow("opening",opening)

cv.imshow("closing",closing)


cv.waitKey(0)

cv.destroyAllWindows()
cv.waitKey(1)#bug en opencv en linux y mac que no deja cerrar las ventanas.  https://github.com/opencv/opencv/issues/7479
cv.waitKey(1)
cv.waitKey(1)


h, w = dilation.shape[:2]

contours0, hierarchy = cv.findContours( dilation.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

def update(levels):
    vis = np.zeros((h, w, 3), np.uint8)
    levels = levels - 3
    cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
        3, cv.LINE_AA, hierarchy, abs(levels) )
    cv.imshow('contours', vis)
update(3)
cv.createTrackbar( "levels+3", "contours", 3, 7, update )
cv.imshow('image', img)
cv.waitKey()
cv.destroyAllWindows()

print(1)