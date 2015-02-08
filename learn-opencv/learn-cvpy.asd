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

Image blending
------------------------------------
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

Image transformations 
-------------------------------------
Scaling
import numpy as np
img = cv2.imread('messi5.jpg')
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#OR
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

Translation
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

Rotation
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

Affine Transformation
img = cv2.imread('drawing.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

Perspective Transform
img = cv2.imread('sudokusmall.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

Color Conversions
------------------------------------------
*  Take each frame of the video
*  Convert from BGR to HSV color-space
*  We threshold the HSV image for a range of blue color
*  Now extract the blue object alone, we can do whatever on that image we want.
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

Image Thresholding 
----------------------------------------
import cv2
import numpy as np
img = cv2.imread('gradient.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)


2D Filter 
----------------------------------------
Convolution
img = cv2.imread('opencv_logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

Averaging
blur = cv2.blur(img,(5,5))

Gaussian Blurring
blur = cv2.GaussianBlur(img,(5,5),0)

Laplacian and Sobel filters
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

Canny Edge Detection
img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

Pyramids
---------------------------------------
A = cv2.imread('apple.jpg')
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
get_ipython().magic(u'logstart detectoy.ipy append')
get_ipython().magic(u'logstop')
get_ipython().system(u'ls -F --color ')
get_ipython().system(u'rm -i detectoy.ipy')
get_ipython().magic(u'logstart learn-cvpy.asd append')
import cv2
get_ipython().system(u'ls -F --color ')
get_ipython().magic(u'logstop')
