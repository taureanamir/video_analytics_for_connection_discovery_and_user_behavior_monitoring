import os
import re

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

if not os.path.exists("Caffe_Files"):
    os.makedirs("Caffe_Files")

try:
    os.remove("Caffe_Files/full_database.txt")
except OSError:
    pass

final_label = 0
tmp_label = 0
full_database_file= open("Caffe_Files/full_database.txt", "w+")
for i in range(0,len(second_level)):
        name = second_level[i].rsplit('/', 1)[-1]
        try:
            label = re.search('Label(.+?)Frame', name).group(1)
        except AttributeError:
        # Label and Frame not found in the original string
            label = '0' # apply your error handling
        label2 = int(label)
        if label2 != tmp_label:
            tmp_label=label2
            final_label+=1
	if 'Label' in second_level[i]: # In order not to have problematic files like .DS_STORE in the database
	        full_database_file.write(second_level[i]+" "+str(final_label)+"\n")

full_database_file.close()
