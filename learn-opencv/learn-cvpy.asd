Accessing and Modifying pixel values
------------------------------------
>>> import cv2
>>> import numpy as np
>>> img = cv2.imread('messi5.jpg')
>>> px = img[100,100]
>>> print px
[157 166 200]
# accessing only blue pixel
>>> blue = img[100,100,0]
>>> print img.shape
(342, 548, 3)
>>> print img.size
562248
>>> print img.dtype
uint8

Image ROI
>>> ball = img[280:340, 330:390]
>>> img[273:333, 100:160] = ball

Splitting and Merging Image Channels
>>> b,g,r = cv2.split(img)
>>> img = cv2.merge((b,g,r))


