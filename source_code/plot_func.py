import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import configuration parameters
import config as cfg

from binary_image import unwarper
from curve_computation import measure_curvature, vehicle_offset_calc

# Annotate a frame image
def annotate_frame(undist, left_line, right_line):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)
    # Get left_fit and right_fit
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Calc polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarper(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_fit = np.polyfit(ploty * cfg.ym_per_pix, left_fitx * cfg.xm_per_pix, 2)
    right_fit = np.polyfit(ploty * cfg.ym_per_pix, right_fitx * cfg.xm_per_pix, 2)
    # Compute curvature for the right and left lines
    left_curve, right_curve = measure_curvature(left_fit, right_fit, ploty)
    # Compute vehicle offset relative to the middle of the lane
    vehicle_offset = vehicle_offset_calc(undist, left_fitx[-1], right_fitx[-1])

    # plot curvature info
    label_str = 'Left line radius: %.1f km' % (left_curve/1000)
    result = cv2.putText(result, label_str, (15, 45), 0, 1, (255, 255, 255), 2)
    label_str = 'Right line radius: %.1f km' % (right_curve/1000)
    result = cv2.putText(result, label_str, (15, 85), 0, 1, (255, 255, 255), 2)

    # Annotate vehicle offset from center
    if vehicle_offset <= 0:
        label_str = 'Vehicle offset to the left: %.1f m' % (abs(vehicle_offset))
    else:
        label_str = 'Vehicle offset to the right: %.2f m' % (abs(vehicle_offset))
    result = cv2.putText(result, label_str, (15, 130), 0, 1, (255, 255, 255), 2)

    return result

# This function is used to plot images before and after a particular pipeline step
def plot_images(img_before, img_after, binary_img_flag, title_before, title_after, fontsize, file_name = 'processed', *args):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    f.tight_layout()

    ax1.imshow(img_before)
    ax1.set_title(title_before, fontsize=fontsize)

    if binary_img_flag:
        ax2.imshow(img_after,'gray')
        #Plots the left and right polynomials on the lane lines
        ax2.plot(args[0][0], args[0][2], color='yellow')
        ax2.plot(args[0][1], args[0][2], color='yellow')
    else:
        ax2.imshow(img_after)

    ax2.set_title(title_after, fontsize=fontsize)

    # Show the figure if needed
    if cfg.plot_figures:
        plt.show()
    # Store the figure if needed
    if cfg.store_img:
        plt.savefig(cfg.output_img_folder + file_name + '.jpg')


