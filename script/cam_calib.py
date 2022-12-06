#!/usr/bin/env python

import numpy as np
import cv2
import time
cv2.namedWindow("Camera Calibration")
cv2.moveWindow("Camera Calibration", 800, 0)
video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

prev_frame_time = time.time()
cal_image_count = 0
frame_count = 0

while (True):
    ret, frame = video_capture.read()
    frame_count += 1
    if frame_count == 30:
        pwd_add = '/home/ammar/sitl/cam_pose_pkg/images/'
        cv2.imwrite(str(pwd_add) + "cal_image_"+str(cal_image_count)+".jpg", frame)
        cal_image_count += 1
        frame_count = 0
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame,
                "FPS:" + str(int(fps)),
                (10, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (100, 255, 0),
                2,
                cv2.LINE_AA)
    cv2.imshow("Camera Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
