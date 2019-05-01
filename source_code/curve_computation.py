import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

# Import configuration parameters
import config as cfg


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, buf_len = 5):
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=buf_len)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = deque(maxlen=buf_len) # circular buffer
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False]), np.array([False]), np.array([False])]

# This function is a reuse from the lecture. Finds lane pixels
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // cfg.nwindows)
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
    for window in range(cfg.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - cfg.margin
        win_xleft_high = leftx_current + cfg.margin
        win_xright_low = rightx_current - cfg.margin
        win_xright_high = rightx_current + cfg.margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > cfg.minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > cfg.minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# This function is a reuse from the lecture with minor modification to pass challenge video.
# Fits a second order polynomial.
def fit_polynomial(binary_warped, left_line, right_line):

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Checks that the lines are separated by approximately the same distance horizontally and
        # compares poly coefficients with previous fits. If it fails this frame is discarded
        # Perform this check only for the challenge video, don't do it for project video or test images
        is_a_good_frame = ((np.abs(left_fitx[-1] - right_fitx[-1] - (left_fitx[0] - right_fitx[0])) < cfg.th) & \
         (np.abs(left_fit[0] - left_line.current_fit[0]) < cfg.th1) \
         & (np.abs(left_fit[1] - left_line.current_fit[1]) < cfg.th2) & \
         (np.abs(left_fit[2] - left_line.current_fit[2]) < cfg.th3))
        # Check against maximum lane width
        is_a_good_frame &= (np.abs(left_fitx[-1] - right_fitx[-1]) < cfg.lane_max_width) & \
                           (np.abs(left_fitx[0] - right_fitx[0]) < cfg.lane_max_width)

        #if (0 == cfg.video_mode) | (cfg.video_file_name == '../project_video') | is_a_good_frame:
        if (0 == cfg.video_mode) | is_a_good_frame:
            ## Visualization ##
            # Colors in the left and right lane regions
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            left_line.recent_xfitted.append(left_fitx)
            left_line.best_fit.append(left_fit)
            right_line.recent_xfitted.append(right_fitx)
            right_line.best_fit.append(right_fit)
        else:
            print('bad frame')
            #pass  # skip this 'bad' frame

    except:
        print('bad frame')
        #pass

    return out_img, left_line, right_line

# Sets the poly coefficients to the last coefficients computed
def long_term_filter_init(left_line, right_line):
    left_line.bestx = left_line.recent_xfitted[-1]
    right_line.bestx = right_line.recent_xfitted[-1]
    left_line.current_fit = left_line.best_fit[-1]
    right_line.current_fit = right_line.best_fit[-1]
    return left_line, right_line

# Takes a mean over accumulated over time poly coefficients
def long_term_filter(left_line, right_line):
    left_line.bestx = np.mean(left_line.recent_xfitted, axis=0)
    right_line.bestx = np.mean(right_line.recent_xfitted, axis=0)
    left_line.current_fit = np.mean(left_line.best_fit, axis=0)
    right_line.current_fit = np.mean(right_line.best_fit, axis=0)
    return left_line, right_line

# Calculate the radius of curvature in meters for both lane lines
def measure_curvature(left_fit_cr, right_fit_cr, ploty):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve in meters (radius of curvature)
    left_curverad = ((1 + (left_fit_cr[0] * 2 * y_eval * cfg.ym_per_pix + left_fit_cr[1]) ** 2) ** (3 / 2)) / np.abs(
        2 * left_fit_cr[0])

    # Calculation of the left line here
    right_curverad = ((1 + (right_fit_cr[0] * 2 * y_eval * cfg.ym_per_pix + right_fit_cr[1]) ** 2) ** (3 / 2)) / np.abs(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad

# Calculate vehicle center offset in meters
def vehicle_offset_calc(undist, bottom_x_left, bottom_x_right):
    # Calculate vehicle center offset in pixels
    vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
    # Convert pixel offset to meters
    vehicle_offset *= cfg.xm_per_pix

    return vehicle_offset

# Fits a second order polynomial to each line. Reuse from a lecture
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    #Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit #
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty, left_fit, right_fit

#  Search for the new line within +/- some margin around the old line center.
def search_around_poly(binary_warped, left_line, right_line):
    margin = cfg.search_around_poly

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                    left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                    right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Fit new polynomials
    try:
        left_fitx, right_fitx, ploty, left_fit, right_fit = \
            fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        # Checks that the lines are separated by approximately the same distance horizontally and
        # compares poly coefficients with previous fits. If it fails this frame is discarded
        # Perform this check only for the challenge video, don't do it for project video or test images
        is_a_good_frame = ((np.abs(left_fitx[-1] - right_fitx[-1] - (left_fitx[0] - right_fitx[0])) < cfg.th) & \
                           (np.abs(left_fit[0] - left_line.current_fit[0]) < cfg.th1) \
                           & (np.abs(left_fit[1] - left_line.current_fit[1]) < cfg.th2) & \
                           (np.abs(left_fit[2] - left_line.current_fit[2]) < cfg.th3))
        # Check against maximum lane width
        is_a_good_frame &= (np.abs(left_fitx[-1] - right_fitx[-1]) < cfg.lane_max_width) & \
                       (np.abs(left_fitx[0] - right_fitx[0]) < cfg.lane_max_width)


        if is_a_good_frame:
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                            ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                             ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Store coefficients into a circular buffer
            left_line.recent_xfitted.append(left_fitx)
            right_line.recent_xfitted.append(right_fitx)
            left_line.best_fit.append(left_fit)
            right_line.best_fit.append(right_fit)
        else:
            print('bad frame')
            #pass  # skip this 'bad' frame

    except:
        print('bad frame')
        #pass
    return out_img, left_line, right_line




