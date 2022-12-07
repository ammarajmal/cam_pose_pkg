#!/usr/bin/env python

import numpy as np
import cv2
import time
frame_per_second = 30
cv2.namedWindow("Camera Calibration")
cv2.moveWindow("Camera Calibration", 800, 0)
video_capture = cv2.VideoCapture(2)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, frame_per_second)

prev_frame_time = time.time()
cal_image_count = 0
frame_count = 0
copyFrame = None
print('Press "Space" to save an image, "Enter" to start camera calibration, and  "Esc" or "q" to quit')

while (True):
    ret, frame = video_capture.read()
    copyFrame = frame.copy()

    # 1. Save an image(Space Key) -- if we see a valid checkerboard image
    # 2. Start Camera clibration (Enter Key) -- if we wanted to start camera calibration so long as we have enough images like 10 or 15
    # 3. Exit(Escape Key)

    inputKey = cv2.waitKey(frame_per_second)
    
    # find chessboard corners and draw then on the frame 
    ret, corners = cv2.findChessboardCorners(frame, (9,6), None)
    if ret == True:
        cv2.drawChessboardCorners(frame, (9,6), corners, ret)
    cv2.imshow("Camera Calibration", frame)
    

    if inputKey == ord(' '):
        print('-- Space pressed... saving image #'+str(cal_image_count)+'.jpg')
        cv2.imwrite("cal_image_"+str(cal_image_count)+".jpg", copyFrame)
        cal_image_count += 1
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
        # cv2.imshow("Camera Calibration", copyFrame)
    elif inputKey == 13:
        print('-- Enter pressed... Starting Camera Calibaration')

    elif inputKey == ord('q') or inputKey == 27:
        print('-- Quitting...')
        break
video_capture.release()
cv2.destroyAllWindows()
