import time
import cv2
import os
import argparse
import sys

def video_to_frames(file_or_dir, input_dir, output_dir):
    """Function to extract frames from input video file and save them as separate frames in an output directory.

    Args:
        input_dir: Input video file.
        output_dir: Output directory to save the frames.

    Returns:
        None
    """
    if file_or_dir == 'file':
        filename = os.path.basename(input_dir)
        new_output_dir = os.path.join(output_dir, filename.split('.')[0])

        # print("Filename: ", filename)
        # print("Output Dir:",  new_output_dir)

        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)

        time_start = time.time()
        video_capture = cv2.VideoCapture(filename)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("----------------------------------------------------------------")
        print("Video to frames conversion for: ", filename)
        print("----------------------------------------------------------------")
        print("Find the output frames at: ", new_output_dir)
        print("----------------------------------------------------------------")
        print ("Number of frames: ", video_length)
        count = 0
        print ("Converting video..\n")
        while video_capture.isOpened():
            ret,frame = video_capture.read()
            cv2.imwrite(new_output_dir + "/%#06d.jpg" % (count+1), frame)
            count = count + 1
            if (count % 1000 == 0):
                print("%d files written." %count)

            if (count > (video_length-1)):
                time_end = time.time()
                video_capture.release()
                print ("Frame extraction completed.\n%d frames extracted" %count)
                print ("Conversion time: %d seconds." %(time_end-time_start))
                break
    else:
        for filename in os.listdir(input_dir):
            new_output_dir = os.path.join(output_dir, filename.split('.')[0])
            filename = os.path.join(input_dir, filename)

            # print("Filename: ", filename)
            # print("Output Dir:",  new_output_dir)

            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)

            time_start = time.time()
            video_capture = cv2.VideoCapture(filename)
            video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            print("----------------------------------------------------------------")
            print("Video to frames conversion for: ", filename)
            print("----------------------------------------------------------------")
            print("Find the output frames at: ", new_output_dir)
            print("----------------------------------------------------------------")
            print ("Number of frames: ", video_length)
            count = 0
            print ("Converting video..\n")
            while video_capture.isOpened():
                ret,frame = video_capture.read()
                cv2.imwrite(new_output_dir + "/%#06d.jpg" % (count+1), frame)
                count = count + 1
                if (count % 1000 == 0):
                    print("%d files written." %count)

                if (count > (video_length-1)):
                    time_end = time.time()
                    video_capture.release()
                    print ("Frame extraction completed.\n%d frames extracted" %count)
                    print ("Conversion time: %d seconds." %(time_end-time_start))
                    break

def main(args):
    video_to_frames(args.file_or_dir, args.input_dir, args.output_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file_or_dir', type=str, help='Input is a directory or a file.')
    parser.add_argument('input_dir', type=str, help='Input video directory.')
    parser.add_argument('output_dir', type=str, help='Output directory to write frames.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
