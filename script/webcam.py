#!/usr/bin/env python

# *********************
# webcam video stream
# *********************
import cv2
import time
# 0 - Dell Webcam

cap = cv2.VideoCapture('filename.avi')
pTime = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cTime = time.time()
        fps = int(1/(cTime - pTime))
        cv2.putText(frame, f'FPS: {int(fps)}', (420, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 1), 3)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  break
    else:   break
cap.release()
cv2.destroyAllWindows()
    