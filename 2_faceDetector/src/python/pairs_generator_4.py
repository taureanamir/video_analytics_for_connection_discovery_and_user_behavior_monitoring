import os
import sys
import random
import shutil
from collections import defaultdict
import scipy.io as sio


full_db = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/labelThu1Dec2.txt",'r')

print "Wait a second. A big database is being generated, the process may take few minutes!"

labeled = defaultdict(list)

line_number=1
while 1:
    txt = full_db.readline()
    if txt == '':
        break
    source,_,label = txt.rpartition(' ')
    labeled[int(label)].append(line_number)
    line_number+=1

pos_pairs = []
neg_pairs = []
pairs = []
total_length_pairs = 0

#print len(labeled)-1
for k in range(1, len(labeled)-1):
    for j in range(k+1, len(labeled)):
         if labeled[k]:
          if labeled[j]:
            print k
            print j
            pairs.append([labeled[k][0],labeled[j][0]])
            total_length_pairs+=1
            

#print total_length_pairs
dic = {'pair':pairs}

sio.savemat("/home/vgl-gpu/Programs/Nabil/ait-research-study/src/face_id/face_verification_experiment-master/code/homkrun_Thu1Dec_pair2.mat", dic)

full_db.close()
