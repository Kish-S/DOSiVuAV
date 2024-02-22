import cv2
import numpy as np
import glob
import os

h_ROI_factor = 13/20

w_top_left_ROI_factor = 9.5/20
w_top_right_ROI_factor = 5.5/10
w_bot_right_ROI_factor = 8/10
w_bot_left_ROI_factor = 3/12

def warper(img, src, dst):

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep the same size as the input image

    # Draw the source points on the original image
    cv2.polylines(img, [np.int32(src)], isClosed=True, color=(0, 255, 0), thickness=3)

    return warped, img


for path in glob.glob('thresholded/*.jpg'):
    
    #Load thresholded image
    img = cv2.imread(path)


    img_size = (img.shape[1], img.shape[0])
    src = np.float32( 
    [[(img_size[0] * w_top_left_ROI_factor), img_size[1] * h_ROI_factor],
    [(img_size[0] * w_bot_left_ROI_factor), img_size[1]],
    [(img_size[0] * w_bot_right_ROI_factor), img_size[1]],
    [(img_size[0] * w_top_right_ROI_factor), img_size[1] * h_ROI_factor ]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    # Apply the perspective transform
    warped_binary, img_with_src = warper(img, src, dst)

    # Save the images
    img_name = os.path.basename(path)
    cv2.imwrite(f'perspective_transformed/{img_name}', warped_binary)
    cv2.imwrite(f'with_src/{img_name}', img_with_src)

