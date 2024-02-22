import numpy as np
import cv2
import glob
import os

# Load calibrated camera parameters
calibratio = np.load('camera_cal/calib.npz')
mtx = calibratio['mtx']
dist = calibratio['dist']
rvecs = calibratio['rvecs']
tvecs = calibratio['tvecs']
i = 0
for path in glob.glob('test_images/*.jpg'):
    
    img = cv2.imread(path)
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # crop and save the image
    x, y, w, h = roi
    croppedImg = undistortedImg[y:y+h, x:x+w]
    img_name = os.path.basename(path)
    
    cv2.imwrite(f'undistorted/{img_name}', croppedImg)
cv2.destroyAllWindows()