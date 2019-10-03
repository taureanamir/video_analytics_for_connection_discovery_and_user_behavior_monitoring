import os
import cv2

input_dir = '/home/avl/Programs/darknet/data/homkrun/train/'
output_dir = '/home/avl/Programs/darknet/data/homkrun/resized/train/'
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        print(filename)

        img = cv2.imread(input_dir + filename, cv2.IMREAD_UNCHANGED)

        print('Original Dimensions : ', img.shape)

        scale_percent = 50  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_dir + filename, resized_img)


















