import cv2
import numpy as np
import glob
import os

def perform_gradient_thresholding(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Sobel operator to calculate gradients in x direction
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobelx**2)
    # Scale the gradient to 8-bit for thresholding
    scaled_gradient = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    # Apply binary thresholding to create a binary image
    _, binary_image = cv2.threshold(scaled_gradient, 60, 255, cv2.THRESH_BINARY)

    return binary_image

def perform_color_thresholding_white(img):
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    # Create a mask for white color
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    return mask_white

def perform_color_thresholding_yellow(img):
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow color
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    return mask_yellow


for path in glob.glob('undistorted/*.jpg'):
    img = cv2.imread(path)

    gradient_thresh = perform_gradient_thresholding(img)

    white_color_thresh = perform_color_thresholding_white(img)

    yellow_color_thresh = perform_color_thresholding_yellow(img)

    # Combine color and gradient thresholds using bitwise OR
    combined_thresh = cv2.bitwise_or(gradient_thresh, cv2.bitwise_or(white_color_thresh, yellow_color_thresh))

    
    # Save the image
    img_name = os.path.basename(path)
    cv2.imwrite(f'thresholded/{img_name}', combined_thresh)



