#! /bin/bash
# Author: Amir Rajak (taureanamir@gmail.com; st119480@ait.asia)
# Created on: Feb 26, 2019
# Modified On:
# Modification Log:
# Remarks:
# This script moves files from a directory into train and test directories.
# For e.g. dir/sub-dir/file-1.png ... file-n.png
# Output will be dir/file1.png ... file-n.png

# list of folders in array
array=(1 10012 1014 1043 1059 1082 1098 1113 22 2734 349 43 4974 5666 70 80 8016 8032 8049 9008 9040 9056 9072 9088 10 10013 1015 1044 1060 1083 1099 1114 2295 28 35 439 5 57 71 8001 8017 8033 8050 9009 9025 9041 9057 9073 9089 1000 10014 1016 1045 1061 1084 11 1115 23 2842 3561 4475 50 58 7146 8002 8018 8034 8051 9010 9026 9042 9058 9074 9090 10000 10015 1017 1046 1063 1085 1116 29 36 4487 5028 59 7180 8003 8019 8035 8053 9011 9027 9043 9059 9075 9091 10001 1002 1018 1047 1064 1086 1101 12 24 2971 3606 4499 51 6 72 8004 8020 8036 8054 9028 9044 9060 9076 9092 10002 1003 1019 1048 1065 1087 1102 13 2443 3 3688 45 52 60 7244 8005 8021 8037 8055 9013 9029 9061 9077 9093 10003 1004 1020 1049 1067 1088 1103 14 25 30 37 451 61 73 8006 8022 8038 8057 9014 9030 9046 9062 9078 9094 10004 1005 1029 1050 1068 1089 1104 15 26 3035 3713 455 53 62 74 8007 8023 8039 8058 9047 9063 9079 9095 10005 1006 1030 1051 1069 1090 1105 16 2621 3068 38 46 54 63 75 8008 8024 8040 9 9016 9032 9048 9064 9080 9097 10006 1007 1034 1052 1070 1091 1106 17 2626 3075 39 460 5418 64 7544 8009 8025 8041 9001 9033 9049 9065 9081 9098 10007 1008 1035 1053 1071 1092 1107 179 2631 3148 3992 465 5423 65 76 8026 8042 9002 9018 9050 9066 9082 9099 10008 1009 1037 1054 1072 1093 1108 18 2649 32 4 47 55 66 7667 8011 8027 8044 9003 9051 9067 9083 10009 1010 1038 1055 1073 1094 1109 19 2667 3293 40 48 5557 67 77 8012 8028 8045 9004 9020 9052 9068 9084 1001 1011 1039 1056 1074 1095 1110 2 2683 33 489 56 68 78 8013 8029 8046 9005 9037 9053 9069 9085 10010 1012 1041 1057 1080 1096 1111 20 2689 34 41 49 5603 79 8014 8030 9006 9038 9054 9070 9086 10011 1013 1042 1058 1081 1097 1112 21 27 3401 42 4969 5648 7 8 8015 8031 8048 9007 9023 9039 9055 9071 9087 9012 )

array2=(1100 69 9031 9045 5274 9034 9024 9022 9019 8010 8047 2336 9021 9015 9035 9036 4008 9017)
root_dir="/home/gunner/drive/AIT/thesis/dataset/Final-face-dataset/"
train_dir="$root_dir/train"
test_dir="$root_dir/test"
#root_dir="Sorted/"

for i in ${array[@]}
do
  echo "------Root dir i----------------"
  echo "$root_dir$i"
  # echo "--------------------------------------------------------------"
  # echo "final dir with files: "$dir
  # echo " ------------------------------------------ top 5 files -------------------------------------"
  # echo "$top5files"

  echo "Making traninig dir $train_dir"/"$i"
  mkdir -p $train_dir"/"$i

  echo "Making test dir $test_dir"/"$i"
  mkdir -p $test_dir"/"$i

  echo "Moving top 5 files to test dir " $test_dir"/"$i
  echo "cd $root_dir$i && mv `ls | head -5` $test_dir/$i && mv *.* $train_dir/$i && rmdir $root_dir$i"
  cd $root_dir$i && mv `ls | head -5` $test_dir/$i && mv *.* $train_dir/$i && rmdir $root_dir$i
  echo "--------------------------------------------------------------"
done

for i in ${array2[@]}
do
  echo "------Root dir i----------------"
  echo "$root_dir$i"
  # echo "--------------------------------------------------------------"
  # echo "final dir with files: "$dir
  # echo " ------------------------------------------ top 5 files -------------------------------------"
  # echo "$top5files"

  echo "Making traninig dir $train_dir"/"$i"
  mkdir -p $train_dir"/"$i

  echo "Making test dir $test_dir"/"$i"
  mkdir -p $test_dir"/"$i

  echo "Moving top 1 files to test dir " $test_dir"/"$i
  echo "cd $root_dir$i && mv `ls | head -1` $test_dir/$i && mv *.* $train_dir/$i && rmdir $root_dir$i"
  cd $root_dir$i && mv `ls | head -1` $test_dir/$i && mv *.* $train_dir/$i && rmdir $root_dir$i
  echo "--------------------------------------------------------------"
done
