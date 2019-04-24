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

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg.sobel_kernel_size)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= cfg.mag_thresh[0]) & (gradmag <= cfg.mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg.sobel_kernel_size)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= cfg.abs_grad_thresh[0]) & (absgraddir <= cfg.abs_grad_thresh[1])] = 1

    # Return the binary image
    return binary_output

def binary_image(img):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    r_channel = img[:, :, 1]

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x

    gradx = abs_sobel_thresh(l_channel, orient='x')
    #grady = abs_sobel_thresh(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orient='y')
    #mag_binary = mag_thresh(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #dir_binary = dir_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    thresh = (200, 255)
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 200) & (r_channel <= 255)] = 1

    combined = np.zeros_like(gradx)
    #combined[((gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
    combined[((gradx == 1) & (binary == 1))] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= cfg.s_thresh[0]) & (s_channel <= cfg.s_thresh[1])] = 1

    # Stack each channel
    # color_binary = np.dstack((rxbinary, sxbinary, s_binary)) * 255
    color_binary = np.dstack((np.zeros_like(s_binary), np.uint8(binary), np.zeros_like(s_binary))) * 255
    return color_binary






def perspective_transform():
    return



def bird_view(img):
    bin_img = binary_image(img)
    #perspective_transform()
    return bin_img


if __name__ == '__main__':
    output_img_folder = 'output_images/'
    images = glob.glob(output_img_folder + 'undist_test_img*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        bin_img = bird_view(img)

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        f.tight_layout()

        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=40)

        ax2.imshow(bin_img)
        ax2.set_title('Binary Result', fontsize=40)
        plt.show()

