import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import config as cfg

def binary_image(img):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    #r_channel = img[:, :, 1]

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # rxbinary = np.zeros_like(r_channel)
    # rxbinary[(r_channel >= 200) & (r_channel <= 255)] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= cfg.sx_thresh[0]) & (scaled_sobel <= cfg.sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= cfg.s_thresh[0]) & (s_channel <= cfg.s_thresh[1])] = 1

    #sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    #abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    #s_channel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= cfg.sx_thresh[0]) & (s_channel <= cfg.sx_thresh[1])] = 1

    # Stack each channel
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
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

