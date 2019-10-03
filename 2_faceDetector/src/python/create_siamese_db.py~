import os
import sys
import random
import shutil
from collections import defaultdict



val = open("//home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/labelCasiaValMNIST.txt",'r')
val1 = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/CasiaVal1onMNIST.txt",'w+')
val2 = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/CasiaVal2onMNIST.txt",'w+')


print "Wait a second. A big database is being generated, the process may take few minutes!"

labeled2 = defaultdict(list)

while 1:
    txt = val.readline()
    if txt == '':
        break
    source,_,label = txt.rpartition(' ')
    labeled2[int(label)].append(source)


for k in range(0, len(labeled2)-1):
    for i in range(0, len(labeled2[k])-1):
        for j in range(i+1, len(labeled2[k])-1):
            val1.write(labeled2[k][i]+' 1\n')
            val2.write(labeled2[k][j]+' 1\n')
            val1.write(labeled2[k][i]+' 0\n')
            while True:
                rd = random.choice(range(1,i-1) + range(i+1,len(labeled2)-1))
                if labeled2[rd] != []:
                    break
            val2.write(labeled2[rd][random.randint(0,len(labeled2[rd])-1)]+' 0\n')

val.close()
val1.close()
val2.close()


# Same for the test file.

#train = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/test.txt",'r')
#train1 = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/test1.txt",'w+')
#train2 = open("/home/vgl-gpu/Programs/Nabil/ait-research-study/Caffe_Files/test2.txt",'w+')

#labeled = defaultdict(list)

#while 1:
 #   txt = train.readline()
  #  if txt == '':
   #     break
   # source,_,label = txt.rpartition(' ')
   # labeled[int(label)].append(source)


#for k in range(0, len(labeled)-1):
 #   for i in range(0, len(labeled[k])-1):
  #      for j in range(i+1, len(labeled[k])-1):
   #         train1.write(labeled[k][i]+' 1\n')
    #        train2.write(labeled[k][j]+' 1\n')
     #       train1.write(labeled[k][i]+' 0\n')
      #      while True:
       #         rd = random.choice(range(1,i-1) + range(i+1,len(labeled)-1))
        #        if labeled[rd] != []:
         #           break
          #  train2.write(labeled[rd][random.randint(0,len(labeled[rd])-1)]+' 0\n')

#train.close()
#train1.close()
#train2.close()
