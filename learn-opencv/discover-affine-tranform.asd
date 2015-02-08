# Create transformed image

cv2.imread('checker.png')
img = cv2.imread('checker.png')
rows,cols,ch = img.shape
print img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
import numpy as np
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('checker_trans.png', dst)


# Apply sift to find features

import cv2
import numpy as np
MIN_MATCH_COUNT = 10
img = cv2.imread('checker.png')
img_trans = cv2.imread('checker_trans.png')
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(img_trans,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
print matches
good = []
for m,n in matches:
        if m.distance < 0.7*n.distance:
                good.append(m)
        
print good
