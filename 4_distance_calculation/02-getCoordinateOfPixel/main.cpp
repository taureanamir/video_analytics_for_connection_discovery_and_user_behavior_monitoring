//#include<opencv2/cv2.h>
//#include<highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void mouseEvent(int event, int x, int y, int flags, void* param)
{
    IplImage* img = (IplImage*) param;
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        printf("%d,%d\n",
        x, y);
    }
}

int main(int argc, char** argv)
{
    // Read image from file
    IplImage *img=cvLoadImage(argv[1],1);

    //Create a window
    cvNamedWindow("My Window", 1);

    //set the callback function for any mouse event
    cvSetMouseCallback("My Window",mouseEvent,&img);

    //show the image
    cvShowImage("My Window", img);

    // Wait until user press some key
    cvWaitKey(0);

    return 0;

}