import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import config as cfg

from camera_calibration import camera_calibration, distortion_correction
from binary_image import binary_image, warper
from plot_func import annotate_frame, plot_images
from curve_computation import fit_polynomial

def pipeline(img, *args):
    undist = distortion_correction(img)
    color_binary, bin_img = binary_image(undist)
    binary_warped = warper(bin_img)
    out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped)
    result = annotate_frame(undist, left_fitx, right_fitx, ploty, left_fit, right_fit)

    if cfg.video_mode == 0:
        if cfg.plot_figures | cfg.store_img:
            # Plot the result
            temp = args[0].split('\\')
            file_name = temp[1].replace('.jpg', '')
            plot_images(img, undist, 0, 'Original Image', 'Undistored Image', 20, file_name + '_undist')
            plot_images(undist, color_binary, 0, 'Undistored Image', 'Binary image, blue - yellow color detection, '
                        'green - white color detection', 10,file_name + '_binary_color')
            plot_images(undist, out_img, 1, 'Undistored Image', 'Binary image in bird view', 20,
                        file_name + '_binary_bird_view', (left_fitx, right_fitx, ploty))
            plot_images(undist, result, 0, 'Undistored Image', 'Processed Image', 20, file_name + 'final')

    return result


if __name__ == '__main__':

    if cfg.compute_calib_params:
        camera_calibration()

    if cfg.video_mode == 0:
        images = glob.glob(cfg.test_img_folder + '*.jpg')
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            pipeline(img, fname)
    else:

        if cfg.store_video == 1:
            if cfg.clip_video:
                clip = VideoFileClip(cfg.video_file_name + '.mp4').subclip(cfg.clip_start, cfg.clip_end)
            else:
                clip = VideoFileClip(cfg.video_file_name + '.mp4')
            processed_video = clip.fl_image(pipeline)
            processed_video.write_videofile('out_' + cfg.video_file_name + str(cfg.num_of_frames) + '_frames.mp4', audio=False)

        else:
            cap = cv2.VideoCapture(cfg.video_file_name + '.mp4')
            out = cv2.VideoWriter('out_' + cfg.video_file_name + str(cfg.num_of_frames) + '_frames.mp4',
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=20.0, frameSize=(1280, 720))

            while cap.isOpened():
                ret, color_frame = cap.read()
                if ret:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                    merged_image = pipeline(color_frame)
                    #cv2.imshow('blend', merged_image*255)
                    cv2.imshow('blend', cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()