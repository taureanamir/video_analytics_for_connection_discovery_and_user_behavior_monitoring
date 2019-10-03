#!/bin/bash

mkdir -p Database;
if [ $# -eq 0 ]; then
	echo "Expected argument: Path to the database.";
	echo "Optional argument: 1 for demo. 0 for default ie no demo"
elif [ $# -eq 1 ]; then
	find $1/ -iname '*.mp4' -exec ./FaceDetector {} 0 \;
elif [ $# -eq 2 ]; then
	find $1/ -iname '*.mp4' -exec ./FaceDetector {} $2 \;
fi
#python src/python/fulldbfile.py
if [ $# -eq 0 ]; then
	echo -e "All the videos have been processed. The generation of the database is complete!\n";
	echo -e "Now you can modify the labels of the file Database/Caffe_Files/full_database.txt. Then just launch ./trainandval.sh\n"
fi
