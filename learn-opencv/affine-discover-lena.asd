get_ipython().magic(u'logstart affine-discover-lena.asd append')
get_ipython().system(u'ls -F --color ')
img = cv2.imread('Lenna.png')
import cv2
cv2.__version__
import cv2
get_ipython().magic(u'logstart affine-discover-lena.asd append')
import cv2
cv2.__version__
img = cv2.imread('Lenna.png')
rows,cols,ch = img.shape
print img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
import numpy as np
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('Lenna-transformed.png',img)
cv2.imwrite('Lenna-transformed.png',dst)
MIN_MATCH_COUNT = 10
img_trans = cv2.imread('Lenna-transformed.png')
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
        
if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
h,w = img.shape
img.shape
m
flann.knnMatch(des1,des2,k=2)
flann.knnMatch(des1,des2,k=2)[0]
asd = flann.knnMatch(des1,des2,k=2)[0]
asd[0]
asd[0]
m,n = flann.knnMatch(des1,des2,k=2)[0]
m
m.distance
print good
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
for i in src_pts.size:
    print src_pts[i], dst_pts[i]
    
for i in range(src_pts.size):
    print src_pts[i], dst_pts[i]
    
for i in range(src_pts.size):
    print src_pts[i]
    
for i in range(src_pts.size):
    print src_pts[i][0]
    
for i in range(src_pts.size):
    print src_pts[i], dst_pts[i]
    
for i in range(len(src_pts)):
    print src_pts[i], dst_pts[i]
    
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
for i in range(len(src_pts)):
    print src_pts[i], dst_pts[i]
    
print m
for m in good:
    print m
    
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
for m in good:
    print m.queryIdx
    
kp1[m.queryIdx]
kp1[m.queryIdx].pt
asd = kp1[m.queryIdx]
asd
asd.angle
asd.class_id
asd.octave
asd.response
asd.size
asd.pt
print img
print img[0]
print img[0][0]
img[m.queryIdx]
asd = kp1[m.queryIdx]
asd
print good
matches = sorted(Ggood, key = lambda x:x.distance)
matches = sorted(good, key = lambda x:x.distance)
matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
img3 = cv2.drawMatches(img,kp1,img_trans,kp2,matches[:10], flags=2)
img3 = cv2.drawMatches(img,kp1,img_trans,kp2,matches[:10], img3)
img3 = cv2.drawMatchesKnn(img,kp1,img_trans,kp2,matches[:10], flags =2 )
img3 = cv2.drawMatchesKnn(img,kp1,img_trans,kp2,matches[:10], None)
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img,kp1,img_trans,kp2,matches[:10], None, **draw_params)
matches
matches[0:10]
for m in matches:
    print kp1[m.queryIdx].pt
    
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
print M
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
M
get_ipython().magic(u'history')
img
rows,cols,ch = img.shape
img_check_affine = cv2.warpAffine(img,M,(cols,rows))
M
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
img_check_affine = cv2.warpPerspective(img,M,(cols,rows))
cv2.imwrite("Leena_check_affine.png", img_check_affine)
get_ipython().magic(u'history')
