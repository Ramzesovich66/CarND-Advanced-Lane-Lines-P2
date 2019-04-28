import numpy as np
import cv2
import matplotlib.pyplot as plt
# Import configuration parameters
import config as cfg

# Defines a function that applies Sobel x or y, then takes an absolute value and applies a threshold.
# 1) Take the derivative in x or y given orient = 'x' or 'y'
# 2) Take the absolute value of the derivative or gradient
# 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
# 4) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
# 5) Return this mask as your binary_output image
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

# Computes and applies perpective transform
def warper(img):
    # Compute perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(cfg.perspective_transform_src, cfg.perspective_transform_dst)
    # Apply perpective transform
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    # Return the result
    return warped

# Computes and applies perpective transform
def unwarper(img):
    # Compute perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(cfg.perspective_transform_dst, cfg.perspective_transform_src)
    # Apply perpective transform
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    # Return the result
    return unwarped


# This is the main function in extracting yellow and white line pixels
def binary_image(img):
    img = np.copy(img)

    # Convert an image into HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Separate the L, and S channels
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Calculate directional gradient in order to extract white line (step 1.1)
    gradx = abs_sobel_thresh(l_channel, orient='x')

    # Yellow line (Step 2.1)
    s_binary = np.zeros_like(s_channel)
    # Perform color thresholding on an S channel to extract yellow line
    s_binary[(s_channel >= cfg.s_thresh[0]) & (s_channel <= cfg.s_thresh[1])] = 1

    # Stack each channel
    if cfg.algo_version == 1:
        # Additionally Extract white line by converting an image into gray scale and then do color thresholding (step 1.2)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, thresh=210, maxval=255, type=cv2.THRESH_BINARY)
        combined_white = np.zeros_like(gradx)
        # Now, 'AND' step 1.1 and step 1.2 for the best white line extraction
        #combined_white[((gradx == 1) & (binary == 255))] = 1
        combined_white[((binary == 255))] = 1
        # Yellow line (step 2.2). Convert an image into hsv color space and do color thresholding
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_min_yellow = np.all(hsv > np.array([0, 100, 100]), axis=2)
        hsv_max_yellow = np.all(hsv < np.array([40, 255, 255]), axis=2)
        hsv_yellow_bin = hsv_min_yellow & hsv_max_yellow
        combined_yellow = np.zeros_like(gradx)
        #combined_yellow[((s_binary == 1) & (hsv_yellow_bin == 1))] = 1
        combined_yellow[((hsv_min_yellow == 1))] = 1
        # Create colorized binary image in order to see yellow and white lines separately from each other
        color_binary = np.dstack((np.zeros_like(s_binary), combined_white, combined_yellow)) * 255
        # Combine both white and yellow lines
        binary_img = combined_white | combined_yellow
    else:
        color_binary = np.dstack((np.zeros_like(s_binary), gradx, s_binary)) * 255
        # Combine both white and yellow lines
        binary_img = gradx | s_binary

    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    if cfg.morphologyex_on:
        kernel = np.ones((3, 3), np.uint8)
        color_binary = cv2.morphologyEx(color_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return color_binary, binary_img

