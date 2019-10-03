# from __future__ import division, print_function, absolute_import
import argparse

import face_recog.face as face

from timeit import time
import warnings
warnings.filterwarnings('ignore')
import sys
import cv2
import numpy as np
from PIL import Image
from tracking_DS_yV3.yolo import YOLO

from tracking_DS_yV3.deep_sort import preprocessing
from tracking_DS_yV3.deep_sort import nn_matching
from tracking_DS_yV3.deep_sort.detection import Detection
from tracking_DS_yV3.deep_sort.tracker import Tracker
from tracking_DS_yV3.tools import generate_detections as gdet
from deepgaze.deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from scipy import misc
import math
import itertools



# def add_overlays(frame, faces, frame_rate):
def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)


def overlay_framerate(frame, frame_rate):
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


# Get the names of the output layers
def getOutputsNames(net):
    # print('--------------------------Get output names ----------------' )
    # print(net)
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def eulerAnglesToRotationMatrix(roll, pitch, yaw):
    x = roll
    y = pitch
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot


def line_to_vector(pt_cust1, pt_headpose, pt_cust2):
    vec1 = pt_headpose - pt_cust1 # vector from customer1 to where s/he's looking at
    vec2 = pt_cust2 - pt_cust1  # vector from customer1 to customer 2
    return vec1, vec2


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radian = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = math.degrees(angle_radian)
    return angle_degree

def get_pairs(frame, list, proximity_threshold, laeo_threshold, larger_group_list):
    # lists to hold group, age of group and num of frames the group is missed
    # if action == 'sitting':
    #     larger_group_list_sit = []
    # else:
    #     larger_group_list_stand = []

    for i in range(len(list) - 1):
        for j in range(i + 1, len(list)):

            st_person_i_position = np.asarray(list[i][2])
            st_person_j_position = np.asarray(list[j][2])

            # np.linalg.norm gives the Frobenius Norm which is also known as Euclidean Norm
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
            distance = np.linalg.norm(st_person_i_position - st_person_j_position)
            # print("Distance between track %s and track %s is %.2f: " %(standing_list[i][0],
            # standing_list[j][0], distance))
            if distance <= proximity_threshold:
                vec_LOS_Pi, vec_Pi_to_Pj = line_to_vector(list[i][4][0],
                                                          list[i][4][1], list[j][4][0])

                vec_LOS_Pj, vec_Pj_to_Pi = line_to_vector(list[j][4][0],
                                                          list[j][4][1], list[i][4][0])

                # print(vec_LOS_Pi, vec_Pi_to_Pj)

                angle_Pi_Pj = angle_between(vec_LOS_Pi, vec_Pi_to_Pj)
                angle_Pj_Pi = angle_between(vec_LOS_Pj, vec_Pj_to_Pi)
                #
                # print("Angles %s, %s", angle_Pi_Pj, angle_Pj_Pi)

                if abs(angle_Pi_Pj) <= laeo_threshold and abs(angle_Pj_Pi) <= laeo_threshold:
                    st_person_i_bbox = list[i][3]
                    st_person_j_bbox = list[j][3]
                    # print("%s. Track %s and Track %s belong to a group. Angle: %s -> %s is %s and "
                    #       "Angle %s -> %s is %s. " % (list[i][1], list[i][0], list[j][0], list[i][0],
                    #                                   list[j][0], str(angle_Pi_Pj), list[j][0],
                    #                                   list[i][0], str(angle_Pj_Pi)))

                    # print("---------------List:---------", list)
                    larger_group_list.append([[list[i][0][0], list[j][0][0]],[st_person_i_bbox],[st_person_j_bbox]])

                    cv2.rectangle(frame, (int(st_person_i_bbox[0]), int(st_person_i_bbox[1])),
                                  (int(st_person_i_bbox[2]), int(st_person_i_bbox[3])), (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(st_person_j_bbox[0]), int(st_person_j_bbox[1])),
                                      (int(st_person_j_bbox[2]), int(st_person_j_bbox[3])), (0, 0, 255), 2)

def get_laeo(frame, img_face, face_bb, is_dummy_face, track_bbox, my_head_pose_estimator_obj ):

    camera_matrix = np.float32([[138.56407, 0.0, 80.0],
                                [0.0, 138.56407, 80.0],
                                [0.0, 0.0, 1.0]])
    # since the cropped image is resized to 160 * 160
    c_x = 80
    c_y = 80

    axis = np.float32([[0.5, 0.0, 0.0],
                       [0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.5]])

    tvec = np.array([0.0, 0.0, 1.0], np.float)  # translation vector
    camera_distortion = np.zeros((4, 1))  # Assuming no lens distortion
    roll = my_head_pose_estimator_obj.return_roll(img_face, radians=True)  # Evaluate the roll angle using a CNN
    pitch = my_head_pose_estimator_obj.return_pitch(img_face, radians=True)  # Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator_obj.return_yaw(img_face, radians=True)  # Evaluate the yaw angle using a CNN

    # print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0, 0, 0]) + "," + str(
    #     pitch[0, 0, 0]) + "," + str(yaw[0, 0, 0]) + "]\n")

    rot_matrix = eulerAnglesToRotationMatrix(-roll[0, 0, 0], -pitch[0, 0, 0], -yaw[0, 0, 0])
    rvec, jacobian = cv2.Rodrigues(rot_matrix)
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

    if is_dummy_face:
        p_start = np.array((int(face_bb[0] + c_x), int(face_bb[1] + c_y)))
        p_stop = np.array((int(face_bb[0] + imgpts[2][0][0]), int(face_bb[1] + imgpts[2][0][1])))
    else:
        p_start = np.array((track_bbox[0] + face_bb[0] + int(c_x),
                            track_bbox[1] + face_bb[1] + int(c_y)))
        p_stop = np.array((track_bbox[0] + face_bb[0] + int(imgpts[2][0][0]),
                           track_bbox[1] + face_bb[1] + int(imgpts[2][0][1])))

    cv2.line(frame, (int(p_start[0]), int(p_start[1])), (int(p_stop[0]), int(p_stop[1])), (0, 0, 255), 3)  # Red
    cv2.circle(frame, (int(p_start[0]), int(p_start[1])), 1, (0, 255, 0), 3)  # Green
    return p_start, p_stop

def merge_pairs(pairs, frame):

    pairs_track_ids = []
    trackId_bbox_dict = {}
    # color codes to map different groups
    color_map = {0: [0,255,255], 1: [255,0,255], 2: [188,188,238], 3: [208,187,114],
                 4: [161,240,145], 5: [255,159,98], 6: [242,194,188]}

    for i in range(len(pairs)):
        pairs_track_ids.append(pairs[i][0])
        trackId_bbox_dict.update([(pairs[i][0][0], pairs[i][1]), (pairs[i][0][1], pairs[i][2])])

    # print("Dictionary ", trackId_bbox_dict)
    merged = [ set(x) for x in pairs_track_ids ] # operate on sets only
    finished = False
    while not finished:
        finished = True
        for a, b in itertools.combinations(merged, 2):
            if a & b:
                # we merged in this iteration, we may have to do one more
                finished = False
                if a in merged: merged.remove(a)
                if b in merged: merged.remove(b)
                merged.append(a.union(b))
                break # don't inflate 'merged' with intermediate results

    for i in range(len(merged)):
        # r =
        color_code = i % 7 # we are using 6 different colors
        merged[i] = list(merged[i])
        color = color_map[color_code]

        for j in range(len(merged[i])):
            trck_bbox = trackId_bbox_dict.get(merged[i][j])

            cv2.rectangle(frame, (int(trck_bbox[0][0]), int(trck_bbox[0][1])),
                          (int(trck_bbox[0][2]), int(trck_bbox[0][3])), (color), 2)

    return merged


def generate_jsonFile (track_dict,sit_pair_set, stand_pair_set ):

    track_list_sorted = list(sorted(track_dict.keys()))

    list_sit_pair = list(sit_pair_set)
    list_stand_pair = list(stand_pair_set)
    json_string = "{\n\t\"nodes\":[\n"

    for i in range(len(track_list_sorted)):
        # if i == 0:
        #
        #     print("\t\t{\"name\":\"node" + str(track_list_sorted[i]) + "\"},")

        if i == (len(track_list_sorted) - 1):
            json_string += "\t\t{\"name\":\"Track " + str(track_list_sorted[i]) + "\"} \n"

        else:
            json_string += "\t\t{\"name\":\"Track " + str(track_list_sorted[i]) + "\"}, \n"

    json_string += "\t],\n" \
                   "\t\"links\":[\n"

    for i in range(len(list_sit_pair)):
        if i == (len(list_sit_pair) - 1):
            json_string += "\t\t{\"source\":" + str(track_list_sorted.index(list_sit_pair[i][0])) + \
                           ", \"target\":" + str(track_list_sorted.index(list_sit_pair[i][1])) + "} \n"
        else:
            json_string += "\t\t{\"source\":" + str(track_list_sorted.index(list_sit_pair[i][0])) + \
                           ", \"target\":" + str(track_list_sorted.index(list_sit_pair[i][1])) + "}, \n"

    for i in range(len(list_stand_pair)):
        if i == (len(list_stand_pair) - 1):
            json_string += "\t\t{\"source\":" + str(track_list_sorted.index(list_stand_pair[i][0])) + \
                           ", \"target\":" + str(track_list_sorted.index(list_stand_pair[i][1])) + "} \n"
        else:
            json_string += "\t\t{\"source\":" + str(track_list_sorted.index(list_stand_pair[i][0])) + \
                           ", \"target\":" + str(track_list_sorted.index(list_stand_pair[i][1])) + "}, \n"

    json_string += "\t]\n" \
                   "}"

    f = open("visualization/graphFile.json", "w")
    f.write(json_string)
    f.close()

def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Initialize the parameters for action recognition
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image

    # Loading action recognition models (Classes: SITTING and STANDING)
    classesFile = "action_recognition/data/behavior-recognition-obj.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    # modelConfiguration = "action_recognition/cfg/behavior-recognition-yolov3-tiny.cfg"
    # modelWeights = "action_recognition/model/behavior-recognition-yolov3-tiny_31000.weights"

    modelConfiguration = "action_recognition/cfg/behavior-recognition-yolov3.cfg"
    modelWeights = "action_recognition/model/4k_yolov3_resizedImg/behavior-recognition-yolov3_4000.weights"

    filename = modelWeights.split(".")[0]
    # print(filename)
    filename = filename[-6:]
    # print(filename)
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Loading Tracking model (deep_sort)
    model_filename = 'tracking_DS_yV3/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Model for head pose estimation
    # head_pose_sess = tf.Session()


    # Output File
    writeVideo_flag = True
    # cam = "cam2/"
    cam = "cam1/"
    # check the homography matrix too, to ensure you use the corresponding homography matrix
    input_file_path = "/mnt/drive/Amir/Thesis/dataset/behavior-recognition/" + cam
    # input_file = "1.mp4"
    input_file = "6.mp4"

    # Draw the predicted bounding box
    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                      (left + round(1.5 * labelSize[0]), top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    # def postprocess(frame, outs):
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    # print(center_x, center_y, width, height, left, top)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        # print("Indices: ", indices)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        return indices, confidences, boxes, classIds

    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    # Video stream to use
    # video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture("rtsp://Admin:1234@192.168.15.181:554/live/stream1")
    video_capture = cv2.VideoCapture(input_file_path + input_file)
    # video_capture = cv2.VideoCapture(cam + input_file_path + input_file)

    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(input_file + filename + '.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    # Make empty white image
    scale = 100
    graph = np.ones((20 * scale, 20 * scale, 3), np.uint8) * 255  # Multiplied by 255 to create a white background
    # image_det = np.ones((12 * scale, 12 * scale, 3), np.uint8) * 255 # Multiplied by 255 to create a white background

    # Cam2 homography matrix
    # homography_matrix = np.float32([[-4.17551031e+01, -3.59557832e+01, 4.84417189e+04],
    #                                 [-5.43485668e+00, 6.36654640e+01, -3.29934621e+04],
    #                                 [-1.36511635e-02, -7.80084395e-02, 1.00000000e+00]])

    # Cam1 homography matrix
    homography_matrix = np.float32([[1.14988815e+00, -8.46877085e-01, -3.20430319e+02],
                                    [-3.29483350e-01, -2.00043462e+00, 1.61263272e+03],
                                    [-6.42556370e-04, 2.05615143e-03, 1.00000000e+00]])

    str_pts_world = ""
    # offsets are used to move the origin coordinates
    offset_X = 600
    offset_Y = 600
    bbox_proximity_threshold = 100
    sittingPersonDistanceThreshold = 200  # these are pixel values which scale to 100px = 2ft in the real world.
    standingPersonDistanceThreshold = 100  # these are pixel values which scale to 100px = 2ft in the real world.

    my_head_pose_estimator = CnnHeadPoseEstimator()  # Head pose estimation object
    laeo_threshold = 45

    track_dict = {}
    sit_pair_set = set()
    stand_pair_set = set()

    while True:
        # Capture frame-by-frame
        # print("************************* Frame start *******************")
        ret, frame = video_capture.read()
        print("\nFrame Number: ", video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        # frameWidth = frame.shape[1]
        # frameHeight = frame.shape[0]

        # print("Frame dimension: ", frameWidth, frameHeight)


        face_recognition_rate = (frame_count % frame_interval)
        # if face_recognition_rate == 0:
        # faces = face_recognition.identify(frame)
        # print ("Num of faces: ",len(faces))

        # Check our current fps
        end_time = time.time()
        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            start_time = time.time()
            frame_count = 0

        overlay_framerate(frame, frame_rate)

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # print("Blob")
        # print(blob)

        # Sets the input to the network
        net.setInput(blob)

        # print('--------------------------  After Sets the input to the network----------------')
        # print(getOutputsNames(net))

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        post_process_indices, confs, action_boxes, class_ids = postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t)
        # and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # ----------------------------------------- Behavior part

        # Tracking in the video starts here
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # list to hold trackid, action and bbox coordinates
        sitting_list = []
        standing_list = []
        sitting_pairs = []
        standing_pairs = []
        sit_large_grp = []
        stand_large_grp = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            centerX = (int(bbox[0]) + int(bbox[2])) / 2
            centerY = (int(bbox[1]) + int(bbox[3])) / 2
            bbox_height = int(bbox[3]) - int(bbox[1])
            bbox_width = int(bbox[2]) - int(bbox[0])
            bbox_bottom_mid_X = centerX
            bbox_bottom_mid_Y = centerY + bbox_height / 2
            # print("Track id: %d" % (track.track_id))
            # print("bbox", bbox)


            negative_trackbox = 0
            for i in bbox:
                if i < 0:
                    negative_trackbox += 1

            if (negative_trackbox > 0):
                continue

            # recognize face in every track bbox
            # if face_recognition_rate == 0:
            croppedTrackBbox = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            if (len(croppedTrackBbox) == 0):
                continue


            # print("Track Bbox", croppedTrackBbox)
            faces = face_recognition.identify(croppedTrackBbox)
            add_overlays(croppedTrackBbox, faces)

            # get face image from the MTCNN detector and feed it to the head pose estimator.
            # If there's no face detected then use the top 20% of the bounding box as a face.
            # if face_recognition_rate == 0:
            if len(faces) == 0:
                dummyFace_bbox = np.zeros(4, dtype=np.float32)
                dummyFace_bbox[0] = bbox[0]
                dummyFace_bbox[1] = bbox[1]
                dummyFace_bbox[2] = bbox[2]
                dummyFace_bbox[3] = bbox[1] + 0.25 * bbox_height
                dummyFace = frame[int(dummyFace_bbox[1]):int(dummyFace_bbox[3]),
                            int(dummyFace_bbox[0]):int(dummyFace_bbox[2])]

                # print("dummyFace: ", dummyFace)
                dummyFace = misc.imresize(dummyFace, (160, 160), interp='bilinear')
                # print("dummyFace after resize: ", dummyFace)
                p_start, p_stop = get_laeo(frame, dummyFace, dummyFace_bbox, True, bbox, my_head_pose_estimator)
            else:
                for i, croppedface in enumerate(faces):
                    # print("--------------------------------Faces from executor--------------------------------")
                    # print(croppedface.image)

                    croppedface_bb = croppedface.bounding_box.astype(int)
                    p_start, p_stop = get_laeo(frame, croppedface.image, croppedface_bb, False, bbox, my_head_pose_estimator)

            iou_list = []
            conf_list = []
            class_id_list = []
            bbox_list = []
            for i in post_process_indices:
                i = i[0]
                action_box = action_boxes[i]

                # the action box that we get here are in format (left, top, left + width, top + height)
                # this center calculation can be moved to the function post_process_indices.
                # here we are calculating the center for every track, while it can be calculated per frame basis.
                action_box = [action_box[0], action_box[1], action_box[0] + action_box[2],
                              action_box[1] + action_box[3]]
                action_box_centerX = (int(action_box[0]) + int(action_box[2])) / 2
                action_box_centerY = (int(action_box[1]) + int(action_box[3])) / 2
                bbox_proximity = np.linalg.norm(
                    np.array([action_box_centerX, action_box_centerY]) - np.array([centerX, centerY]))

                if bbox_proximity <= bbox_proximity_threshold:
                    conf = confs[i]
                    class_id = class_ids[i]
                    # print("Bboxes and conf: ", action_box, bbox,conf, class_id )
                    iou = bb_intersection_over_union(action_box, bbox)
                    iou_list.append(iou)
                    conf_list.append(conf)
                    class_id_list.append(class_id)
                    bbox_list.append(bbox)

            # print("IOU list", iou_list)
            # Filter out the customers whose actions are not classified
            if len(iou_list) >= 1:
                max_iou_idx = np.argmax(iou_list)
                track_action_id = class_id_list[max_iou_idx]
                track_action_conf = conf_list[max_iou_idx]
                track_bbox = bbox_list[max_iou_idx]

                if classes:
                    assert (class_id < len(classes))
                    track_action = '%s' % (classes[track_action_id])

                # converting the image coordinates into homogeneous coordinates
                pts_image = np.array([[bbox_bottom_mid_X], [bbox_bottom_mid_Y], [1.0]])

                # computing homogeneous world coordinates from image coordinates
                pts_world = scale * np.dot(homography_matrix, pts_image)

                # converting homogeneous world coordinates to world cartesian coordinates
                pts_world = pts_world / pts_world[-1]

                if track_action_id == 0:
                    sitting_list.append([[track.track_id], [track_action], [int(pts_world[0]) + offset_X,
                                int(pts_world[1]) + offset_Y], track_bbox, [p_start, p_stop]])
                    track_dict.update([(track.track_id, [int(pts_world[0]) + offset_X,
                                int(pts_world[1]) + offset_Y])])
                else:
                    standing_list.append([[track.track_id], [track_action], [int(pts_world[0]) + offset_X,
                                int(pts_world[1]) + offset_Y], track_bbox, [p_start, p_stop]])
                    track_dict.update([(track.track_id, [int(pts_world[0]) + offset_X,
                                                       int(pts_world[1]) + offset_Y])])

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # if there is more than one person standing then proceed to calculate the distance between the persons
        # standing_list[i][4] is two sets of coordinates representing line coming out of the nose
        # standing_list[i][3] is bounding box coordinates
        # standing_list[i][2] is person/track coordinate position
        # standing_list[i][1] is classified action (sitting, standing)
        # standing_list[i][0] is track id
        # same for sitting list.

        if len(standing_list) > 1:
            get_pairs(frame, standing_list, standingPersonDistanceThreshold, laeo_threshold, standing_pairs)
            # print("standing_pairs: ", standing_pairs)
            if len(standing_pairs) > 1:
                stand_merged_pairs = merge_pairs(standing_pairs, frame)
                for i in stand_merged_pairs:
                    stand_large_grp.append(i)
            else:
                if len(standing_pairs) == 1:
                    stand_large_grp.append(standing_pairs[0][0])

            for i in range(len(standing_pairs)):
                # print("Standing Group: ", standing_pairs[i][0])
                stand_pair_set.add(tuple(standing_pairs[i][0]))

        if len(sitting_list) > 1:
            get_pairs(frame, sitting_list, sittingPersonDistanceThreshold, laeo_threshold, sitting_pairs)
            if len(sitting_pairs) > 1:
                sit_merged_pairs = merge_pairs(sitting_pairs, frame)
                for i in sit_merged_pairs:
                    sit_large_grp.append(i)
            else:
                if len(sitting_pairs) == 1:
                    sit_large_grp.append(sitting_pairs[0][0])

            for i in range(len(sitting_pairs)):
                print("Sitting Group: ", sitting_pairs[i][0])
                sit_pair_set.add(tuple(sitting_pairs[i][0]))


        frame_count += 1
        cv2.imshow('Video', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %f" % (fps))


        # Press Q to stop!
        if cv2.waitKey(1) == 27:
            break

        # print("************************* Frame end *******************")

    #  json file to generate HTML web page of social graph
    generate_jsonFile(track_dict, sit_pair_set, stand_pair_set)


    # When everything is done, release the capture
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    # parser.add_argument('yolo', type=str, default='YOLO()')

    return parser.parse_args(argv)


if __name__ == '__main__':
    yolo = YOLO()  # creating object of class YOLO
    main(parse_arguments(sys.argv[1:]))
