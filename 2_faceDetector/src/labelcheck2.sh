a=1


#for each image file in videos folder
for filee in $1/* ; do
     # echo "----------------- All Files-----------------"
     echo $filee

     #make sure it is image file containing string "Label"
     if echo "$filee" | grep -q "Label"; then
       file1=$(echo $filee | cut -d "/"  -f 10) #extract only file name from path. The number 10 may need to be changed as per the number of slashes '/' in the input path.
       #echo "----------------- File1 -----------------:   $file1"
       labelnumber=$(echo "$filee" | grep -o -P '(?<=Label).*(?=Frame)') #extract the label number - string between Label and Frame
       #echo "----------------- Label Number -----------------:   $labelnumber"
       file2="$1/$labelnumber/$file1" #path to store file to
       #echo "----------------- File2 -----------------:   $file2"
       echo $file2
       echo $filee
       mkdir -p $1/$labelnumber #make dir with labelnumber if it does not exist"
       mv $filee $file2 #copy to specified location
     fi



done
