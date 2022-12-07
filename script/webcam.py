#!/usr/bin/env python

# *********************
# webcam video stream
# *********************
import time
import cv2

# 0 - Dell Webcam

# cap = cv2.VideoCapture('filename.avi')
cap = cv2.VideoCapture(0)
pTime = 0
while (cap.isOpened()):
    cTime = time.time()
    ret, frame = cap.read()
    if ret == True:

        fps = int(1/(cTime - pTime))
        cv2.putText(frame, f'FPS: {int(fps)}', (420, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 1), 3)
        cv2.imshow('Frame', frame)
        pTime = cTime
        print(fps)
        if cv2.waitKey(25) & 0xFF == ord('q'):  break
    else:   break
cap.release()
cv2.destroyAllWindows()
    