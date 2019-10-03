This code extracts the face from video input directly. The faces of same people are automatically placed into separate folders. But you need to go through each folder to verify.  

Steps:
1. Change the path of the cascade.xml file as per your machine's location in the FaceDetector.cpp file.
2. Run ./install.sh. This builds the current faceDetector project.
3. Run ./generate_database.sh. It creates a directory "Database" and saves the extracted faces into the same directory. You can change the extension of the input video file here.
4. Execute labelcheck2.sh to sort the images of individuals into a separate directory.
