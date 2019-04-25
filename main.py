import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import config as cfg
import camera_calibration as cc
import binary_image as bi
import curve_computation as cuc
import utility_funct as ut

def pipeline_image():
    images = glob.glob(cfg.test_img_folder + '*.jpg')
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        undist = cc.distortion_correction(img)
        binary_warped = bi.bird_view(undist)
        out_img, left_fitx, right_fitx, ploty = cuc.fit_polynomial(binary_warped)
        result = ut.annotate_frame(undist, binary_warped, left_fitx, right_fitx, ploty)
        if cfg.store_img:
            write_name = cfg.output_img_folder + 'undist_test_img' + str(idx) + '.jpg'
            cv2.imwrite(write_name, cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

        if cfg.plot_figures:
            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            f.tight_layout()

            ax1.imshow(undist)
            ax1.set_title('Undistored Image', fontsize=20)

            #ax2.imshow(binary_warped)
            #Plots the left and right polynomials on the lane lines
            # ax2.plot(left_fitx, ploty, color='yellow')
            # ax2.plot(right_fitx, ploty, color='yellow')
            # ax2.imshow(out_img)
            ax2.imshow(result)
            ax2.set_title('Bird view', fontsize=20)
            plt.show()

def pipeline_video(img):
    undist = cc.distortion_correction(img)
    binary_warped = bi.bird_view(undist)
    out_img, left_fitx, right_fitx, ploty = cuc.fit_polynomial(binary_warped)
    result = ut.annotate_frame(undist, binary_warped, left_fitx, right_fitx, ploty)
    return result


if __name__ == '__main__':

    if cfg.compute_calib_params:
        cc.camera_calibration()

    if cfg.video_mode == 0:
        pipeline_image()
    else:

        if cfg.store_video == 1:
            if cfg.clip_video:
                clip = VideoFileClip(cfg.video_file_name + '.mp4').subclip(cfg.clip_start, cfg.clip_end)
            else:
                clip = VideoFileClip(cfg.video_file_name + '.mp4')
            processed_video = clip.fl_image(pipeline_video)
            processed_video.write_videofile('out_' + cfg.video_file_name + str(cfg.num_of_frames) + '_frames.mp4', audio=False)

        else:
            cap = cv2.VideoCapture(cfg.video_file_name + '.mp4')
            out = cv2.VideoWriter('out_' + cfg.video_file_name + str(cfg.num_of_frames) + '_frames.mp4',
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=20.0, frameSize=(1280, 720))

            while cap.isOpened():
                ret, color_frame = cap.read()
                if ret:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                    merged_image = pipeline_video(color_frame)
                    #cv2.imshow('blend', merged_image*255)
                    cv2.imshow('blend', cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
