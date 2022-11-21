#!/usr/bin/env python

# *********************
# webcam video stream
# *********************
import cv2

# 0 - Dell Webcam

cap = cv2.VideoCapture(2)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  break
    else:   break
cap.release()
cv2.destroyAllWindows()
    