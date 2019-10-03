for filesss in /home/aniqua/faceDetector/src/Database/ML/*/*/*/* ; do
 mogrify -format png $filesss
done 
for filesss2 in /home/aniqua/faceDetector/src/Database/ML/*/*/*/*.jpg ; do
 rm $filesss2
done 

counter=0
for dir in /home/aniqua/faceDetector/src/Database/ML/* ; do
 subject=$(echo $dir | cut -d "/"  -f 8)
 for dir in /home/aniqua/faceDetector/src/Database/ML/$subject/* ; do
   gender=$(echo $dir | cut -d "/"  -f 9)
   for filedir in /home/aniqua/faceDetector/src/Database/ML/$subject/$gender/* ; do
	 emotion=$(echo $filedir | cut -d "/"  -f 10)
 	 count=0
	 for filedir2 in /home/aniqua/faceDetector/src/Database/ML/$subject/$gender/$emotion/* ; do
            count=$((count+1))
            counter=$((counter+1))
	    file=$(echo $filedir2 | cut -d "/"  -f 11)
	    #date=$(echo $file | cut -d "-"  -f 2)
	    file2=/home/aniqua/faceDetector/src/Database/ML/$subject/$gender/${subject}_${emotion}${count}.png
            echo $filedir2
            echo $file2
 	    mv $filedir2 $file2
	 done 
   done
 done
done

rm -r /home/aniqua/faceDetector/src/Database/ML/*/*/SA
rm -r /home/aniqua/faceDetector/src/Database/ML/*/*/NE
rm -r /home/aniqua/faceDetector/src/Database/ML/*/*/HA
