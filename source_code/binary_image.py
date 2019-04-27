import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import config as cfg

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x'):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel_size))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg.sobel_kernel_size))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= cfg.sxy_thresh[0]) & (scaled_sobel <= cfg.sxy_thresh[1])] = 1

    # Return the result
    return binary_output

def warper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(cfg.perspective_transform_src, cfg.perspective_transform_dst)
    Minv = cv2.getPerspectiveTransform(cfg.perspective_transform_dst, cfg.perspective_transform_src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(cfg.perspective_transform_dst, cfg.perspective_transform_src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return unwarped

def binary_image(img):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # white line
    gradx = abs_sobel_thresh(l_channel, orient='x')

    # yellow line
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= cfg.s_thresh[0]) & (s_channel <= cfg.s_thresh[1])] = 1

    # Stack each channel
    if cfg.algo_version == 1:
        # white line
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, thresh=170, maxval=255, type=cv2.THRESH_BINARY)
        combined_white = np.zeros_like(gradx)
        combined_white[((gradx == 1) & (binary == 255))] = 1

        # yellow line
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= cfg.s_thresh[0]) & (s_channel <= cfg.s_thresh[1])] = 1
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_min_yellow = np.all(hsv > np.array([0, 100, 100]), axis=2)
        hsv_max_yellow = np.all(hsv < np.array([40, 255, 255]), axis=2)
        hsv_yellow_bin = hsv_min_yellow & hsv_max_yellow
        combined_yellow = np.zeros_like(gradx)
        #combined_yellow[((s_binary == 1) & (hsv_yellow_bin == 1))] = 1
        combined_yellow[((hsv_min_yellow == 1))] = 1
        color_binary = np.dstack((np.zeros_like(s_binary), combined_white, combined_yellow)) * 255
        binary_img = combined_white | combined_yellow
    else:
        color_binary = np.dstack((np.zeros_like(s_binary), gradx, s_binary)) * 255
        binary_img = gradx | s_binary

    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    if cfg.morphologyex_on:
        kernel = np.ones((3, 3), np.uint8)
        color_binary = cv2.morphologyEx(color_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return color_binary, binary_img


