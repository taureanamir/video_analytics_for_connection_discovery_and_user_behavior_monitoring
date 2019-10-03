#! /bin/bash
# This script moves files from a sub-directory to the directory above it and retains the sub-directory name.
# For e.g. dir/sub-dir/file-1.png ... file-n.png
# Output will be dir/file1.png ... file-n.png

# list of folders in array
array=(8001 8005 8009 8013 8017 8021 8025 8029 8033 8037 8041 8046 8050 8055 9002 9006 9010 9014 9018 9022 9026 9030 9034 9038 9042 9046 8002 8006 8010 8014 8018 8022 8026 8030 8034 8038 8042 8047 8051 8057 9003 9007 9011 9015 9019 9023 9027 9031 9035 9039 9043 9047 8003 8007 8011 8015 8019 8023 8027 8031 8035 8039 8044 8048 8053 8058 9004 9008 9012 9016 9020 9024 9028 9032 9036 9040 9044 9048 8004 8008 8012 8016 8020 8024 8028 8032 8036 8040 8045 8049 8054 9001 9005 9009 9013 9017 9021 9025 9029 9033 9037 9041 9045 9049)

#array=(1 2 3 4 10)
root_dir="/media/gunner/ADATA HD700/with love/Homkrun-data-from-love/"
#root_dir="Sorted/"

for i in ${array[@]}
do
  echo "$root_dir$i"
  dir=$(ls "$root_dir$i")
  echo "--------------------------------------------------------------"
  echo "final dir with files: "$dir
  echo "Moving files from " $root_dir$i"/"$dir"/" "to" $root_dir$i
  mv "$root_dir$i"/$dir/*.* "$root_dir$i"/
  rmdir "$root_dir$i"/$dir/
  echo "Removed directory $root_dir$i"/"$dir"/". Completed moving files from $root_dir$i"/"$dir"/" to" $root_dir$i
  echo "--------------------------------------------------------------"
done
