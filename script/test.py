#!/usr/bin/env python

# *********************
# webcam video stream
# *********************

import time
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
from geometry_msgs.msg import Pose

logitech = 0
webcam = 2
file_name = 'my_rec.mp4'


# Rotataions
# Check if a matrix is a valid rotation matrix

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped)

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# def pose_est(read_file_name):
#     cap = cv2.VideoCapture(read_file_name)

#     while (cap.isOpened()):
#         st_time = time.time()
#         ret, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         if ret is True:
#             cv2.imshow('Frame', frame)
#             end_time = time.time()
#             fps = 1/(end_time-st_time)
#             print(int(fps))
#             st_time = end_time
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()


def pose_recorded_video(read_file_name):
    print('******************************************')
    print('               inside pose                ')
    print('******************************************')
    marker_size = 100
    camera_width = 640
    camera_height = 480
    camera_frame_rate = 30
    
    with open('camera_calib.npy', 'rb') as f:
        camera_matrix = np.load(f)
        camera_distortion = np.load(f)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    print('camera matrix: {}'.format(camera_matrix))
    print('camera distortion: {}'.format(camera_distortion))

    cap = cv2.VideoCapture(read_file_name)
    # cap.set(2,camera_width)
    # cap.set(4,camera_height)
    # cap.set(5,camera_frame_rate)
    prev_frame_time = time.time()
    while (cap.isOpened()):
        st_time = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(640, 480)


        
        if ret is True:
            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            new_frame_time = time.time()
            fps = 1/(new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.putText(frame, "FPS:" + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()




def video_record_mp4(camera, output_file_name):
    capture = cv2.VideoCapture(camera)
    capture.set(3, 640)
    capture.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_file_name, fourcc, 20.0, (640, 480))

    while (True):
        ret, frame = capture.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            print('\n..finished recording file..\n')
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()

def play_recorded_video(read_file_name):
    cap = cv2.VideoCapture(read_file_name)

    while (cap.isOpened()):
        st_time = time.time()
        ret, frame = cap.read()
        
        if ret is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Frame', frame)
            end_time = time.time()
            fps = 1/(end_time-st_time)
            print(int(fps))

            st_time = end_time
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def fps_calc_vid(file_to_read):
    capture = cv2.VideoCapture(file_to_read)
    pre_time_frame = time.time()
    # new_time_frame = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        new_time_frame = time.time()
        fps = 1/(new_time_frame - pre_time_frame)
        pre_time_frame = new_time_frame
        fps = int(fps)
        cv2.putText(frame, "FPS: "+str(fps), (4,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,  )
        cv2.imshow('video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

        
        
        

if __name__ == '__main__':
    # video_record_mp4(logitech, file_name)
    
    # video_record_mp4(webcam,file_name)
    
    # play_recorded_video(file_name)
    # pose_recorded_video(file_name)
    
    fps_calc_vid(file_name)
    
    
    
    