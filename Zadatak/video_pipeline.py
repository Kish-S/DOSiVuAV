import cv2
import numpy as np
from thresholding import perform_color_thresholding_yellow, perform_color_thresholding_white, perform_gradient_thresholding
from perspective_transform import warper
from lane_detection import find_lane_pixels, measure_curvature_real, measure_vehicle_position, warp_lane_boundaries
import os

h_ROI_factor = 13/20
w_top_left_ROI_factor = 9.5/20
w_top_right_ROI_factor = 5.5/10
w_bot_right_ROI_factor = 8/10
w_bot_left_ROI_factor = 3/12

class LaneHistory:
    def __init__(self):
        self.left_fits = []
        self.right_fits = []


    def update_history(self, left_fit, right_fit):
        # Keep a fixed-size history (e.g., last 5 frames)
        history_size = 10
        self.left_fits.append(left_fit)
        self.right_fits.append(right_fit)
        if len(self.left_fits) > history_size:
            self.left_fits.pop(0)
            self.right_fits.pop(0)

    def get_smoothed_coefficients(self):
        # Average coefficients over the history
        left_fit_avg = np.mean(self.left_fits, axis=0)
        right_fit_avg = np.mean(self.right_fits, axis=0)
        return left_fit_avg, right_fit_avg



def is_deviation_strong(left_fit, right_fit, lane_history, threshold):
    # Check if the deviation from the average of the last few frames is strong
    avg_left_fit, avg_right_fit = lane_history.get_smoothed_coefficients()
    left_deviation = np.mean(np.abs(left_fit - avg_left_fit))
    right_deviation = np.mean(np.abs(right_fit - avg_right_fit))

    return left_deviation > threshold or right_deviation > threshold

# Needed to have variant without plotting
def fit_polynomial(img):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, _ = find_lane_pixels(img)

    # If no lines are detected in the current frame, use the history
    if len(leftx) == 0 or len(rightx) == 0 :
        left_fit_smoothed, right_fit_smoothed = lane_history.get_smoothed_coefficients()

        # Generate x values
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit_smoothed[0] * ploty**2 + left_fit_smoothed[1] * ploty + left_fit_smoothed[2]
        right_fitx = right_fit_smoothed[0] * ploty**2 + right_fit_smoothed[1] * ploty + right_fit_smoothed[2]

        return left_fitx, right_fitx, ploty

    # Fit a second-order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if is_deviation_strong(left_fit, right_fit, lane_history, 150):
        left_fit_smoothed, right_fit_smoothed = lane_history.get_smoothed_coefficients()

        # Generate x values
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit_smoothed[0] * ploty**2 + left_fit_smoothed[1] * ploty + left_fit_smoothed[2]
        right_fitx = right_fit_smoothed[0] * ploty**2 + right_fit_smoothed[1] * ploty + right_fit_smoothed[2]

        return left_fitx, right_fitx, ploty
    

    # Update the lane history
    lane_history.update_history(left_fit, right_fit)

    # Smooth coefficients over the history
    left_fit_smoothed, right_fit_smoothed = lane_history.get_smoothed_coefficients()

    # Generate x values
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    
    return left_fitx, right_fitx, ploty

def undistort(img):
    calibration = np.load('camera_cal/calib.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    rvecs = calibration['rvecs']
    tvecs = calibration['tvecs']

    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # crop and save the image
    x, y, w, h = roi
    croppedImg = undistortedImg[y:y+h, x:x+w]
    
    return croppedImg


def thresholding(img):
    gradient_thresh = perform_gradient_thresholding(img)

    white_color_thresh = perform_color_thresholding_white(img)

    yellow_color_thresh = perform_color_thresholding_yellow(img)

    # Combine color and gradient thresholds using bitwise OR
    combined_thresh = cv2.bitwise_or(gradient_thresh, cv2.bitwise_or(white_color_thresh, yellow_color_thresh))

    return combined_thresh


def perspective_transformation(img):
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
    warped_binary, _ = warper(img, src, dst)
    return warped_binary


def perform_lane_detection_and_transform_back(img, undistorted):

    left_fitx, right_fitx, ploty = fit_polynomial(img)
    
    # Measure curvature
    left_curveread, right_curveread = measure_curvature_real(left_fitx, right_fitx, ploty)
    
    # Measure vehicle position
    vehicle_position = measure_vehicle_position(left_fitx, right_fitx, ploty, img.shape[1])

    lcr = int(left_curveread)
    rcr = int(right_curveread)
    vp = "{:.2f}".format(abs(vehicle_position))

    result = warp_lane_boundaries(undistorted, img, left_fitx, right_fitx, ploty)

    textLCR = f'Left curvature radius: {lcr}m'
    textRCR = f'Right curvature radius: {rcr}m'
    textVP =  f'Vehicle position: {vp}m from center'

    cv2.putText(result, textLCR, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, textRCR, (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, textVP, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return result


def process_image(img):

    undistorted = undistort(img)
    thresholded = thresholding(undistorted)
    perspective_transformed = perspective_transformation(thresholded)
    result = perform_lane_detection_and_transform_back(perspective_transformed, undistorted)

    return result


output_folder = 'output_videos'
os.makedirs(output_folder, exist_ok=True)

# Iterate over all video files in the test_videos folder
for video_filename in os.listdir('test_videos'):
    # Initialize the lane history
    lane_history = LaneHistory()
    if video_filename.endswith('.mp4'):
        # Open the input video
        video_capture = cv2.VideoCapture(os.path.join('test_videos', video_filename))

        while(video_capture.isOpened()):
            ret, frame = video_capture.read()
            if ret==True:

                # write the flipped frame
                processed = process_image(frame)


                cv2.imshow('Processed video',processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release resources for the current video
        video_capture.release()
        cv2.destroyAllWindows()