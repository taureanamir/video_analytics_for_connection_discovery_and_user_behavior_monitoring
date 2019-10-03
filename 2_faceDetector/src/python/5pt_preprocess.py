import os
import re
import dlib
from skimage import io
from pymatbridge import Matlab

predictor_path = "src/others/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



first_level_in_database = []
second_level = []
k=0

# Name of the directories in the array named first_level_in_database
for (dirpath, dirnames, filenames) in os.walk("Database"):
    first_level_in_database.extend(dirnames)
    break
first_level_in_database=sorted(first_level_in_database)
# Name of the file in each directory
for i in range(0,len(first_level_in_database)):
    for root, dirs, files in os.walk(os.path.abspath("Database/"+first_level_in_database[i])):
        for file in sorted(files):
            second_level.append(os.path.join(root, file))

if not os.path.exists("Database_5pt"):
    os.makedirs("Database_5pt")

size = 144*144
mlab = Matlab(matlab='/Applications/MATLAB_R2015a.app/bin/matlab')
mlab.start()

for i in range(0,len(second_level)):
        name = second_level[i].split("Database/",1)[1].rsplit(".",1)[0]
        address, name2 = os.path.split(name)
        if name2 != ".DS_Store" and name2 != '':
            if not os.path.exists('Database_5pt/'+address):
                os.makedirs('Database_5pt/'+address)
            pt = open("Database_5pt/"+address+'/'+name2+".5pt","w+")
            img=io.imread(second_level[i])
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                shape = predictor(img, d)
                left_eye = shape.part(45)
                right_eye = shape.part(36)
                pt.write(str(left_eye.x)+' '+str(left_eye.y)+'\n')
                pt.write(str(right_eye.x)+' '+str(right_eye.y)+'\n')
                nose = shape.part(30)
                pt.write(str(nose.x)+' '+str(nose.y)+'\n')
                left_mouse = shape.part(54)
                right_mouse = shape.part(48)
                pt.write(str(left_mouse.x)+' '+str(left_mouse.y)+'\n')
                pt.write(str(right_mouse.x)+' '+str(right_mouse.y)+'\n')
                pt.close()
                ec_y = shape.part(36).y-shape.part(45).y
                ec_y= (ec_y,-ec_y)(ec_y<0)
                mc_y = shape.part(54).y-shape.part(48).y
                mc_y =(mc_y,-mc.y)[ec_y<0]
                ec_mc_y = mc_y-ec_y
                pt.close()
                #computations
                mlab.run_func('src/software/face_db_align.m', {'face_dir': 'Database', 'ffp_dir': 'Database_5pt', 'ec_mc_y': ec_mc_y, 'ec_y': ec_y, 'img_size': size, 'save_dir': 'Database_preprocessed'})