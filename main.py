import numpy as np
import cv2
import glob
#  import importlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import config as cfg
import camera_calibration as cc
import binary_image as bi






def pipeline_image():
    images = glob.glob(cfg.test_img_folder + '*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        undist = cc.distortion_correction(img)
        if cfg.store_img:
            write_name = cfg.output_img_folder + 'undist_test_img' + str(idx) + '.jpg'
            cv2.imwrite(write_name, undist)
        bin_img = bi.binary_image(undist)

        if cfg.plot_figures:
            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            f.tight_layout()

            ax1.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
            ax1.set_title('Undistored Image', fontsize=20)

            ax2.imshow(bin_img)
            ax2.set_title('Binary Result', fontsize=20)
            plt.show()

def pipeline_video():
    clip = VideoFileClip(cfg.video_file_name + '.mp4').subclip(23, 24)
    processed_video = clip.fl_image(bi.binary_image)
    processed_video.write_videofile('out_' + cfg.video_file_name + str(cfg.num_of_frames) + '_frames.mp4', audio=False)


if __name__ == '__main__':

    if cfg.compute_calib_params:
        cc.camera_calibration()

    if cfg.video_mode == 0:
        pipeline_image()
    else:
        pipeline_video()
