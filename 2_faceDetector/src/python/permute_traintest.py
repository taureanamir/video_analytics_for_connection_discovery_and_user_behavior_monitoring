import os
import sys
import random
import shutil

number_of_input_per_label = dict() # is a dictionnary

lines1 = open('Caffe_Files/train1.txt','r')
lines2 = open('Caffe_Files/train2.txt','r')

concat = open('/tmp/concat', 'w+')

while 1:
	line1=lines1.readline().rstrip('\n')
	line2=lines2.readline().rstrip('\n')
	if line1=='':
		break
	concat.write(line1+'@'+line2+'\n')

with open('/tmp/concat','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('/tmp/concat2','w') as target:
    for _, line in data:
        target.write( line )

lines1.close()
lines2.close()

concat2 = open('/tmp/concat2', 'r')
flines1= open('Caffe_Files/train1.txt', 'w+')
flines2= open('Caffe_Files/train2.txt', 'w+')

while 1:
	txt = concat2.readline()
	if txt =='':
		break
	line1,_,line2 = txt.rpartition('@')
	flines1.write(line1+'\n')
	flines2.write(line2)

concat.close()
flines1.close()
flines2.close()
	