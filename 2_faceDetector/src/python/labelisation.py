import os
import sys
import random
import shutil

full_db = open("../../Caffe_Files/labelHomKrunFinalSelectionTestSet.txt", "r")

particular_labels = []
lines_labelised_full_db = []
maximum = 0

while 1:
	txt = full_db.readline()
	if txt == '':
		break
	source,_,label = txt.rpartition(' ')
	name = source.rsplit('/', 1)[-1]
	if name[0].isdigit():
		lines_labelised_full_db.append(source+' '+name[0])
		if int(name[0]) > maximum:
			maximum = int(name[0])
		if not(int(name[0]) in particular_labels):
			particular_labels.append(int(name[0]))
	else:
		lines_labelised_full_db.append(source+' '+label)
		if int(label) > maximum:
			maximum = int(label)

#print particular_labels

# Labels that are manually taken have to be updated if they were previously existing in fulldbfile.py
# After this loop, each line which contains a "particular" label (created manually) will be modified.

current_label = 0
maximum-=1

for i in range(0, len(lines_labelised_full_db)-1):
	source,_,label = lines_labelised_full_db[i].rpartition(' ')
	name = source.rsplit('/', 1)[-1]
	if not(name[0].isdigit()) and (int(label) in particular_labels): 
		if int(label) != current_label:
			maximum+=1
			current_label = int(label)
		lines_labelised_full_db[i] = source+' '+str(maximum)+'\n'

# Let's remove the labels appearing only one time. They are irrelevant.

counter = [0] * (len(lines_labelised_full_db)-1)

for i in range(0, len(lines_labelised_full_db)-1):
	_,_,label = lines_labelised_full_db[i].rpartition(' ')
	counter[int(label)]+=1

for i in range(0, len(lines_labelised_full_db)-1):
	_,_,label = lines_labelised_full_db[i].rpartition(' ')
	if counter[int(label)]<=1:
		lines_labelised_full_db[i] = ''

#print lines_labelised_full_db

# Let's recreate the labeling after all these modifications.

lines_final = []
actual_label = -1
previous_label = -1
for i in range(0, len(lines_labelised_full_db)-1):
	if lines_labelised_full_db != '':
		#print lines_labelised_full_db
		source,_,label = lines_labelised_full_db[i].rpartition(' ')
		name = source.rsplit('/', 1)[-1]
		if name!='':
			if name[0].isdigit():
				lines_final.append(lines_labelised_full_db[i])
			else:
				if previous_label != int(label):
					actual_label+=1
					previous_label = int(label)
				while (actual_label in particular_labels):
					actual_label+=1
				lines_final.append(source+' '+str(actual_label))

#Let's copy the final array in the appropriate file

labelised_full_db = open("../../Caffe_Files/labelHomKrunFinalSelectionTestSet2.txt",'w+')

for i in range(0, len(lines_final)-1):
			labelised_full_db.write(lines_final[i]+"\n")

labelised_full_db.close()
full_db.close()
