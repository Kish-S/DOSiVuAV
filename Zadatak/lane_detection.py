import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension



def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second-order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    fig, ax = plt.subplots()

    # Visualization
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    ax.plot(left_fitx, ploty, color='red')
    ax.plot(right_fitx, ploty, color='blue')
    ax.imshow(binary_warped, cmap='gray')
    
    return fig, left_fitx, right_fitx, ploty, out_img

def measure_curvature_real(left_fitx, right_fitx, ploty):

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)

    # Calculate the radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curverad, right_curverad

def measure_vehicle_position(left_fitx, right_fitx, ploty, image_width):
    # Calculate the position of the vehicle with respect to the center
    lane_center_pixel = (left_fitx[-1] + right_fitx[-1]) / 2
    center_of_image = image_width / 2
    vehicle_position = (lane_center_pixel - center_of_image) * xm_per_pix

    return vehicle_position

def warp_lane_boundaries(undistorted, binary_warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lane lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h_ROI_factor = 13/20
    w_top_left_ROI_factor = 9.5/20
    w_top_right_ROI_factor = 5.5/10
    w_bot_right_ROI_factor = 8/10
    w_bot_left_ROI_factor = 3/12

    # Recast x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    img_size = (undistorted.shape[1], undistorted.shape[0])
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
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)  # Assuming you have 'src' and 'dst' defined from the perspective transform
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    return result

for path in glob.glob('perspective_transformed/*.jpg'):
    binary_warped = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fig, left_fitx, right_fitx, ploty, out_img = fit_polynomial(binary_warped)

    # Measure curvature
    left_curveread, right_curveread = measure_curvature_real(left_fitx, right_fitx, ploty)
    
    # Measure vehicle position
    vehicle_position = measure_vehicle_position(left_fitx, right_fitx, ploty, binary_warped.shape[1])

    # Print curvatures and vehicle position on plot
    lcr = int(left_curveread)
    rcr = int(right_curveread)
    vp = "{:.2f}".format(abs(vehicle_position))
    plt.text(400, 100, f'Left curvature radius: {lcr}m \n Right curvature radius: {rcr}m', color='green')
    plt.text(400, 200, f'Vehicle position: {vp}m from center', color='green')

    # Save detected lanes
    img_name = os.path.basename(path)
    plt.savefig(f'lane_boundary_detected/{img_name}')
    plt.close(fig)
    
    undistorted = cv2.imread(f'undistorted/{img_name}')

    # Warp the detected lane boundaries back onto the original image
    result = warp_lane_boundaries(undistorted, binary_warped, left_fitx, right_fitx, ploty)

    textLCR = f'Left curvature radius: {lcr}m'
    textRCR = f'Right curvature radius: {rcr}m'
    textVP =  f'Vehicle position: {vp}m from center'

    cv2.putText(result, textLCR, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, textRCR, (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, textVP, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    cv2.imwrite(f'warped_back/{img_name}', result)



cv2.destroyAllWindows()
