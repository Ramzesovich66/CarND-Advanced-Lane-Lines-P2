import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
# Import configuration parameters
import config as cfg

from camera_calibration import camera_calibration, distortion_correction
from binary_image import binary_image, warper
from plot_func import annotate_frame, plot_images
from curve_computation import fit_polynomial, Line, search_around_poly, long_term_filter_init, long_term_filter

frame_number = 0

# Create left and right line objects of class Line to keep track of the characteristics of each line detection
left_line = Line(buf_len=cfg.num_of_frames)
right_line = Line(buf_len=cfg.num_of_frames)


# Processing pipeline
def pipeline(img, *args):
    global left_line, right_line, frame_number

    # 1. Apply a distortion correction to raw images.
    undist = distortion_correction(img)
    # 2. Use color transforms, gradients, etc., to create a thresholded binary image.
    color_binary, bin_img = binary_image(undist)
    # 3. Apply a perspective transform to rectify binary image ("birds-eye view").
    binary_warped = warper(bin_img)

    # 4. Detect lane pixels and fit to find the lane boundary.
    # if it is a very first frame or a test image or a video mode without 'search from Prior' enabled
    if ((frame_number == 0) | (0 == cfg.video_mode) | ((1 == cfg.video_mode) & (0 == cfg.apply_search_around_poly))):
        # 4.1 locate the Lane Lines (with histogram and sliding window) and fit a 2nd order polynomial
        out_img, left_line, right_line = fit_polynomial(binary_warped, left_line, right_line)

        if (1 == cfg.video_mode) & (frame_number > 0):
            # 4.2.1 Filter 2nd order polynomial coefficients over several frames
            left_line, right_line = long_term_filter(left_line, right_line)
        else:
            # 4.2.2 If it is a first frame or a test image do not filter polynomial coefficients
            left_line, right_line = long_term_filter_init(left_line, right_line)
    else:
        # 4.1 Perform 'search from Prior' on a video image, i.e. search in a margin around the previous line position
        out_img, left_line, right_line = search_around_poly(binary_warped, left_line, right_line)
        # 4.2 Filter 2nd order polynomial coefficients over several frames
        left_line, right_line = long_term_filter(left_line, right_line)

    # 5. Annotate a video frame with lane detected
    # 5.1 Determine the curvature of the lane and vehicle position with respect to center.
    # 5.2 Warp the detected lane boundaries back onto the original image.
    # 5.3 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    result = annotate_frame(undist, left_line, right_line)
    frame_number += 1

    # 6. If it is a test imaged store/plot images from different pipeline steps (depends on config flags)
    if 0 == cfg.video_mode:
        if cfg.plot_figures | cfg.store_img:
            # Plot the result
            temp = args[0].split('\\')
            file_name = temp[1].replace('.jpg', '')
            ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
            plot_images(img, undist, 0, 'Original Image', 'Undistored Image', 20, file_name + '_undist')
            plot_images(undist, color_binary, 0, 'Undistored Image', 'Binary image, blue - yellow color detection, '
                                                                     'green - white color detection', 10,
                        file_name + '_binary_color')
            plot_images(undist, out_img, 1, 'Undistored Image', 'Binary image in bird view', 20,
                        file_name + '_binary_bird_view', (left_line.bestx, right_line.bestx, ploty))
            plot_images(undist, result, 0, 'Undistored Image', 'Processed Image', 20, file_name + '_final')

    return result


# Main function
if __name__ == '__main__':

    # 1. If needed compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    if cfg.compute_calib_params:
        camera_calibration()

    # 2. A test image processing
    if 0 == cfg.video_mode:
        images = glob.glob(cfg.test_img_folder + '*.jpg')
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            pipeline(img, fname)
    else:
        # 3. Processing a video frame

        # 3.1 Store a annotated video frame into a file
        if 1 == cfg.store_video:
            if cfg.clip_video:
                clip = VideoFileClip(cfg.video_file_name + '.mp4').subclip(cfg.clip_start, cfg.clip_end)
            else:
                clip = VideoFileClip(cfg.video_file_name + '.mp4')
            processed_video = clip.fl_image(pipeline)
            temp = cfg.video_file_name
            processed_video.write_videofile(cfg.output_video_folder + 'out_' + temp[3::] + '_' +
                                            str(cfg.num_of_frames) + '_frames.mp4', audio=False)

        else:
            # 3.2 Display annotated video on a fly without storing it into a file
            cap = cv2.VideoCapture(cfg.video_file_name + '.mp4')
            while cap.isOpened():
                ret, color_frame = cap.read()
                if ret:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                    merged_image = pipeline(color_frame)
                    # cv2.imshow('blend', merged_image*255)
                    cv2.imshow('blend', cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
