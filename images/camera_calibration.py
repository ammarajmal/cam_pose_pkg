import numpy as np
import cv2 as cv
import glob

############################        FIND CHESSBOARD CORNERS - objPonints, imgPoints ############################
chessboard_size = (9, 6)
framesize = (640, 480)

# termination criteria 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)  # 3D points in real world space   # 9x6 = 54   54x3 = 162  162x1 = 162   
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)  # 2D points in image plane        # 9x6 = 54   54x2 = 108  108x1 = 108

# Array to store object points and image points from all the images
objPoints = []  # 3d points in real world space
imgPoints = []  # 2d points in image plane

images = glob.glob('*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgPoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboard_size, corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(500)
cv.destroyAllWindows()


############################        CALIBRATION ############################
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, framesize, None, None)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, framesize, None, None)


print("Camera Calibrated: ", ret, '\n')
print("Camera Matrix: ", mtx, '\n')
print("Distortion Coefficients: ", dist, '\n')
print("Rotation Vectors: ", rvecs, '\n')
print("Translation Vectors: ", tvecs, '\n')


############################        UNDISTORTION ############################
img = cv.imread('cal_image_56.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) 

# undistort
dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)
#  crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_undistort.png', dst)

# Undistort with Remapping
mapx,mapy = cv.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_undistort_remapping.png', dst)


# Reprojection Error
mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objPoints)) )
print("\n\n\n")  
