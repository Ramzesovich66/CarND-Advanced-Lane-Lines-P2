
# System settings
video_mode = 1  # =0: test images, =1 video playback
store_img = 0  # =1: store intermediate images in test images mode
plot_figures = 1  # if set =1 then plots figure at different stages of the pipeline

# Video file to test
video_file_name = 'project_video'
clip_video = 0  # [default =0] Full video playback
                # [= 1] video file will be clipped with the following 2 parameters
clip_start = 0  # presenting the start of the subclip, used when clip_video = 1
clip_end = 30  # presenting the start of the subclip, used when clip_video = 1

# Calibration related params
compute_calib_params = 0  # if set to 1 then calibration params recomputed
x_dim = 9  # chessboard size x axis
y_dim = 6  # chessboard size y axis

#
num_of_frames = 1  # Averaging is done over this number of frames

# Path to the subfolders
cam_cal_folder = 'camera_cal/'  # calibration images
test_img_folder = 'test_images/'  # test images
output_img_folder = 'output_images/'  # output images

# Algo settings
# Binary image settings
algo_version = 0  #  [= 0] simple algo: S color channel (HLS) && Sobel x, gradient threshold
                  #  [= 1] as =0 but additionally performs HSV and gray
if algo_version == 0:
    s_thresh = (170, 255)  # Threshold S color channel (HLS) (170, 255) vs (70, 255)
else:
    s_thresh = (70, 255)

# Gradient algos
sobel_kernel_size = 7
sxy_thresh = (20, 200)  # Sobel x or y, gradient threshold
mag_thresh = (60, 200)  # gradient magnitude threshold
abs_grad_thresh = (0.7, 1.3)  # Absolute value of the gradient direction

