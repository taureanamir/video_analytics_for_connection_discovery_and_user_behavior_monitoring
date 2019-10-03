import cv2
import numpy as np

# img = cv2.imread("images_for_homography_computation/cam2.png")  # queryimage
img = cv2.imread("images_for_homography_computation/cam1.png")  # queryimage

# query_pts refer to the point in the query image that we use to find the homography. We get the query points
# manually using c++ program in this case.
# train_pts are the target points in our 2D plane that correspond to the query_pts in the query image
# we use these 2 sets of points to calculate the ground plane homography of homkrun coffee shop


# query_pts1 = np.float32([[820, 395], [1145, 400], [1075, 610], [665, 575]])
# print(query_pts1)

# The query points are the points of the tiles on the floor, Each tile is a
# 2ft X 2ft square. I've used the scaling of 2ft = 100 px

scale = 100
# For cam2 homography
query_pts = np.float32([[820, 395], [1145, 400], [1075, 610], [665, 575]]).reshape(-1, 1, 2)
train_pts = np.float32([[0,3 * scale], [3 * scale,3 * scale], [3 * scale,0], [0,0]]).reshape(-1, 1, 2)

# For cam1 homography
# query_pts = np.float32([[669, 530], [909, 510], [1033, 636], [778, 678]]).reshape(-1, 1, 2) # cam1.png
# train_pts = np.float32([[0,2 * scale], [2 * scale,2 * scale], [2 * scale,0], [0,0]]).reshape(-1, 1, 2)

print(query_pts)
print("------------------------------------")
print(train_pts)
homographyMatrix, mask = cv2.findHomography(query_pts, train_pts,0)
# print(type(homographyMatrix))
print("Homography Matrix: ")
print(homographyMatrix)
# print(mask)
matches_mask = mask.ravel().tolist()
print(matches_mask)

"""
Homography Matrix: Cam1
[[ 1.14988815e+00, -8.46877085e-01, -3.20430319e+02],
 [-3.29483350e-01, -2.00043462e+00,  1.61263272e+03],
 [-6.42556370e-04,  2.05615143e-03,  1.00000000e+00]]

Homography Matrix: Cam2
[[-4.17551031e+01 , -3.59557832e+01 , 4.84417189e+04],
 [-5.43485668e+00 ,  6.36654640e+01 ,-3.29934621e+04],
 [-1.36511635e-02 , -7.80084395e-02 , 1.00000000e+00]]
"""
