import cv2
import numpy as np

h = 300
w = 300

# create blank image
img = np.zeros((h,w,3), np.uint8)
cv2.line(img,(0,0),(200,200),(255,0,0),5)

cv2.imshow('Color image', img)
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()