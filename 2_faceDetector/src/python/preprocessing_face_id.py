import os
import re
import dlib
import sys, math
from PIL import Image
from skimage import io

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  
  #offset_h = offset_pct[0]
  #offset_v = offset_pct[1]
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image


predictor_path = "/home/vgl-gpu/Programs/Nabil/ait-research-study/src/others/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



first_level_in_database = []
second_level_in_database = []
second_level = []
k=0

# Name of the directories in the array named first_level_in_database
for (dirpath, dirnames, filenames) in os.walk("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrun/HomKrun-Final-Selection2"):
    first_level_in_database.extend(dirnames)
    
    
first_level_in_database=sorted(first_level_in_database)

# Name of the directories in the array named second_level_in_database
for i in range(0,len(first_level_in_database)):
	for (dirpath, dirnames, filenames) in os.walk("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrun/HomKrun-Final-Selection2/"+first_level_in_database[i]):
    		second_level_in_database.extend(dirnames)
    	
	second_level_in_database=sorted(second_level_in_database)
	print second_level_in_database
	
	for j in range(0,len(second_level_in_database)):
		for root, dirs, files in os.walk(os.path.abspath("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrun/HomKrun-Final-Selection2/"+first_level_in_database[i]+"/"+second_level_in_database[j])):
			for file in sorted(files):
				
				second_level.append(os.path.join(root, file))
# Name of the file in each directory


if not os.path.exists("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrunPreprocessed"):
  os.makedirs("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrunPreprocessed")

for i in range(0,len(second_level)):
        name = second_level[i].split("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrun/HomKrun-Final-Selection2/",1)[1]
        if name.rsplit('/', 1)[-1] != ".DS_Store" and name.rsplit('/', 1)[-1] != "":
          if not os.path.exists("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrunPreprocessed/"+name.rsplit('/',1)[0]):
            os.makedirs("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrunPreprocessed/"+name.rsplit('/',1)[0])
          
          try:
            image = io.imread(second_level[i])
            img = Image.open(second_level[i])
            dets = detector(image, 1)
            for k, d in enumerate(dets):
              shape = predictor(image, d)
              left_eye = shape.part(36)
              left_eye = [left_eye.x,left_eye.y]
              right_eye = shape.part(45)
              right_eye = [right_eye.x,right_eye.y]            
              CropFace(img, eye_left=left_eye, eye_right=right_eye, offset_pct=(0.20,0.20), dest_sz=(144,144)).save("/home/vgl-gpu/Programs/Nabil/ait-research-study/DatabaseHomKrunPreprocessed/"+name)
          except:
            pass


