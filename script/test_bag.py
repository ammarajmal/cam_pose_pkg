#!/usr/bin/env python3
import numpy as np
import cv2
import math
import time 
import cv2.aruco as aruco 
import rospy
from geometry_msgs.msg import Pose

# ------------------------------
#  Pose Estimation Calculations
# ------------------------------

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
        
    

marker_size = 100

with open('camera_calib.npy', 'rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)

camera_width = 640
camera_height = 480
camera_frame_rate = 30


cap.set(2,camera_width)
cap.set(4,camera_height)
cap.set(5,camera_frame_rate)
prev_frame_time = time.time()



# --------------------
#  ROS Publishing code
# --------------------

# pose_message.x, pose_message.y, pose_message.theta = 0, 0, 0

def publisher():
    pub = rospy.Publisher('pose', Pose, queue_size=1)
    rospy.init_node("Pose_Publisher", anonymous=True)
    rate = rospy.Rate(10) # Hz
    
    while not rospy.is_shutdown():
        pose_message = Pose()
        
        ret, frame = cap.read()     # grab a frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # find all the aruco markers in the image
        corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners)
            
            # get pose of all single markers
            rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
            
            rvec = rvec_list_all[0][0]
            tvec = tvec_list_all[0][0]
            
            cv2.drawFrameAxes(frame, camera_matrix, camera_distortion, rvec, tvec, 100)
            
            rvec_flipped = rvec * -1
            tvec_flipped = tvec * -1
            rotation_matrix, jacobian = cv2.Rodrigues(rvec_flipped)
            realworld_tvec = np.dot(rotation_matrix, tvec_flipped)
            
            pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)
            
            or_x = realworld_tvec[0]
            or_y = realworld_tvec[1]
            or_z = math.degrees(yaw)
            # tvec_str = "x=%4.0f y=%4.0f z=%4.0f"%(realworld_tvec[0], realworld_tvec[1], math.degrees(yaw))
            tvec_str = "x=%4.0f y=%4.0f z=%4.0f"%(or_x, or_y, or_z)
            cv2.putText(frame, tvec_str, (20,460), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)

            pose_message.position.x = or_x
            pose_message.position.y = or_y
            pose_message.position.z = or_z
            
            # make sure the quaternion is valid and normalized
            pose_message.orientation.x = 0.0
            pose_message.orientation.y = 0.0
            pose_message.orientation.z = 0.0
            pose_message.orientation.w = 1.0
            
            cv2.imshow('frame', frame)
            
        # new_frame_time = time.time()
        # fps = 1/(new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # cv2.putText(frame, "FPS:" + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2, cv2.LINE_AA)

        
            
        
        
        
        
        
        
        pub.publish(pose_message)
        rate.sleep()
        
if __name__ == "__main__":
    try:
        print('so far so good.! :-)')
        publisher()
        
        cv2.destroyAllWindows()
    except rospy:
        pass
        


print('so far so good.! :-)')
