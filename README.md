## Lane Finding Project
 
 
 
---
 
**Summary**
 
The goal of this project is to detect lane-lines on image/video. This was done using the following steps:
 
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use white and yellow color thresholding combined with horizontal gradients, to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Apply this on each frame of a video
 
[//]: # (Image References)
 
[image0]: ./Zadatak/camera_cal/calibration1.jpg "Original_Chess"
[image1]: ./Zadatak/undistorted_chess/calibration1.jpg "Undistorted_Chess"
[image2]: ./Zadatak/undistorted/test1.jpg "Road Transformed"
[image3]: ./Zadatak/thresholded/test1.jpg "Binary Example"
[image4]: ./Zadatak/perspective_transformed/test1.jpg "Warp Example"
[image5]: ./Zadatak/lane_boundary_detected/test1.jpg "Fit Visual"
[image6]: ./Zadatak/warped_back/test1.jpg "Output"
[image7]: ./Zadatak/with_src/test1.jpg "With_Src"
 
---
 
 
### Camera Calibration
 
The code for this step is contained in the first code located in "./examples/example.py".  
 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
 
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
 
![alt text][image0]
 
![alt text][image1]
 
### Pipeline (single images)
 
#### 1. Distortion correction
 
Using the same method for undistortion as for chessboard image and changing input file to `test_1.jpg`, the following result happened:
 
![alt text][image2]
 
#### 2. Thresholding
 
I used a combination of color and gradient thresholds to generate a binary image. Yellow and White color thresholding was performed in functions `perform_color_thresholding_yellow(img)` and `perform_color_thresholding_white(img)` in file `thresholding.py`. Color thresholding was done in *HSV* color space, and as a result a binary mask of yellow and white pixels is returned. Those masks are then combined using `cv2.bitwise_or()` function. Additionaly, horizontal gradient thresholding was applied. The horizontal gradient output(binary) is then once again combined with `cv2.bitwise_or()` with the previus combination of yellow and white masks. 
 
The result can be seen on the following image:
 
![alt text][image3]
 
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
 
The code for my perspective transform includes a function called `warper()`, which appears in file `Zadatak/perspective_transform.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
 
```python
h_ROI_factor = 13/20
w_top_left_ROI_factor = 9.5/20
w_top_right_ROI_factor = 5.5/10
w_bot_right_ROI_factor = 8/10
w_bot_left_ROI_factor = 3/12
 
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
```
 
I verified that my perspective transform was working as expected by drawing the `src` points onto a thresholded image and its warped counterpart to conclude that the region of interest is valid.
 
Thresholded image with src points drawn:
 
![alt text][image7]
 
 
Warped image:
 
![alt text][image4]
 
 
 
#### 4. Lane-line pixel identification and fitting their positions with a polynomial
 
In order to identify lane-line pixels, `find_lane_pixels(binary_warped)` was implemented in file `lane_detection.py`. 
 
First, I took the warped picture and created histogram of the bottom half. Then i found peaks of the left and right halves of the histogram and initialized sliding windows. Windows are iterated through vertically, resulting in detecting nonzero pixels within each window and their indices. If enough pixels are found, recenter the next window based on their mean x-coordinate. Afterwards, concatenation of pixel indices obtained from all windows is done, creating a list of pixels for each lane. Finally, x and y coordinates for each pixel is returned for left and right lane.
 
 
After detecting lane-line pixels, I have performed a 2nd order polynomial fit in function `fit_polynomial(binary_warped)` in file `lane_detection.py` and as a result, drew lane lines on the image.
 
The image after drawing lane-lines:
 
![alt text][image5]
 
#### 5. The radius of curvature of the lane and the position of the vehicle with respect to center
 
The radius of curvature of the lane is calculated in function `measure_curvature_real(left_fitx, right_fitx, ploty)`. It is done using the fitted left and right lane-lines from previous step. I have hardcoded distance conversions from pixels to meter in the following manner:
 
ym_per_pix = 30/720   meters per pixel in y dimension
xm_per_pix = 3.7/700  meters per pixel in x dimension
 
 
Then, fitting the new polynomials to x,y in world space has been done. Finally, radius of the new curvature was calculated.
 
 
Position of the vehicle with respect to center was calculated in function `measure_vehicle_position(left_fitx, right_fitx, ploty, image_width)` in file `lane_detection.py`
 
Output can be seen on the previous image, as text.
 
#### 6. Result plotted back down onto the road.
 
After doing all calculation, in function warp_lane_boundaries(undistorted, binary_warped, left_fitx, right_fitx, ploty) warping the detected lane-lines back onto the original undistorted image is done. The area between them is colored in green color. 
 
The resulting image of lane detection can be seen here:
 
![alt text][image6]
 
---
 
### Pipeline (video)
 
#### 1. Video
 
In order to smooth out parts of the video where there are no lines detected(weak color, in shadow, badly drawn) history buffer of left and right fits was introduced. In the case when there are no lines detected on the current frame either on left or on right side, average of last 10 frame lines are taken and that line is used as lane-line. Additionally, deviation from average of last 10 lines was calculated and if it is too high (it is not a road curve, but noise induced jitter), average of last 10 lines is taken as well.
 
Here's a [link to my video result](https://www.youtube.com/watch?v=4Sa_WUMbAyQ)
 
---
 
### Discussion
 
First major challenge was determining the type of thresholding that i want to use. My though process was: Lines can be either yellow or white, so naturally thresholding with those colors in HSV color space came to mind. Also at first, i had tried to use both horizontal and vertical gradient thresholding, but I quickly figured out that vertical gradient does not help with detecting vertical lines, so i kept only the horizontal part, and was happy with the result.
 
One of the harder challenges was to determince source region of interest points. It took a lot of trial and error, but this result is what i liked the most.
 
Final hard challenge for me was that in video `challenge_02.mp4` my pipeline broke completely as there were frames under the bridge where the left boundary of the lane couldn't be found. Then i thought of an idea to have a history of lines in past N frames, and use them as a fallback, if something is really off in the frame.
 
A challenge that i still did not overcome is in video `challenge_03.mp4` pipeline is not working well. I guess that more complex techniques need to be applied, as there is reflection, motorist and very strong curvatures.