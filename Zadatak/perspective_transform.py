import cv2
import numpy as np
import glob
import os

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

for path in glob.glob('thresholded/*.jpg'):
    
    #Load thresholded image
    img = cv2.imread(path)


    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    # Apply the perspective transform
    warped_binary = warper(img, src, dst)

    # Save the images
    img_name = os.path.basename(path)
    cv2.imwrite(f'perspective_transformed/{img_name}', warped_binary)    

cv2.destroyAllWindows()