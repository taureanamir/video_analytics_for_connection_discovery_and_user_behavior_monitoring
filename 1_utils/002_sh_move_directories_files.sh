#! /bin/bash
# This script moves files from a sub-directory to the directory above it and retains the sub-directory name.
# For e.g. dir/sub-dir/file-1.png ... file-n.png
# Output will be dir/file1.png ... file-n.png

# list of folders in array
array=(441 502 1485 3148 2626 1350 1508 1394 1279 745 3096 2660 460 2971 3637 2131 3040 3533 3168 1515 725 3700 643 489 2171 2154 3003 915 349 1122 2950 1327 3674 1504 2631 1034 1688 3210 2946 490 2676 3713 465 1351 2212 395 455 977 1388 2649 2147 700 956 1343 1539 1718 2811 969 3121 2213 1478 3520 527 1134 1418 2671 2163 899 359 2152 2733 1244 1883 3128 2345 2751 1127 2628 3051 1398 1543 1112 1819 1978 3102 1751 1601 1656 532 2203 3013 627 3112 2725 1170 1078 3082 1514 612 2208 2639 1149 2887 356 1973 1182 3483 1708 1762 707 1367 1007 2988 495 451 1079 1619 277 1875 1467 1295 614 492 587 2522 3150 2124 934 638 110 1064 1184 1962 284 2222 1889 984 1357 634 2955 1337 3617 3155 504 1141 1698 1535 1298 2195 1555 1093 3109 3031 2832 2967 3290 2523 1222 2142 1179 475 2223 2645 2721 1667 2765 2793 1299 3510 1856 1611 1524 1105 1120 2219 503 1500 1192 1060 1082 2799 439 3535 2520 2113 1423 952 3534 2417 1624 1288 2496 1360 139 1815 3234 3539 1533 1546 1012 2724 1371 1736 2635 1851 1430 79 2216 1322 1043 1342 2465 294 1593 2170)

#array=(1 2 3 4 10)
root_dir="/home/gunner/drive/AIT/thesis/dataset/Database/192.168.15.180/"
dest_dir="/home/gunner/drive/AIT/thesis/dataset/Database/"
#root_dir="Sorted/"

for i in ${array[@]}
do
  echo "$root_dir$i"
  dir=$(ls "$root_dir$i")
  echo "--------------------------------------------------------------"
  echo "final dir with files: "$dir
  echo "Moving files from " $root_dir$i"/" "to" $dest_dir
  mv "$root_dir$i/" "$dest_dir/"
  j=i+10011
  echo "Rename directory $dest_dir$i to $dest_dir$j".

  mv $dest_dir$i $dest_dir$j
  # Completed moving files from $root_dir$i"/"$dir"/" to" $root_dir$i
  echo "--------------------------------------------------------------"
done
