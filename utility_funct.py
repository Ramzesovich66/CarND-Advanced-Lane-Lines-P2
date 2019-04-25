import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import binary_image as bi
import curve_computation as cuc

def annotate_frame(undist, left_fitx, right_fitx, ploty, left_fit, right_fit):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = bi.unwarper(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Compute curvature for the right and left lines
    left_curve, right_curve = cuc.measure_curvature_real(left_fit, right_fit, ploty)
    # Compute vehicle offset relative to the middle of the lane
    vehicle_offset = cuc.vehicle_offset_calc(undist, left_fitx[-1], right_fitx[-1])

    # Annotate lane curvature values
    label_str = 'Left line radius: %.1f km' % (left_curve/1000)
    result = cv2.putText(result, label_str, (15, 45), 0, 1, (0, 0, 0), 2)
    label_str = 'Right line radius: %.1f m' % (right_curve/1000)
    result = cv2.putText(result, label_str, (15, 85), 0, 1, (0, 0, 0), 2)

    # Annotate vehicle offset from center
    if vehicle_offset <= 0:
        label_str = 'Vehicle offset to the left: %.1f m' % (abs(vehicle_offset))
    else:
        label_str = 'Vehicle offset to the right: %.2f m' % (abs(vehicle_offset))
    result = cv2.putText(result, label_str, (15, 130), 0, 1, (0, 0, 0), 2)

    return result
