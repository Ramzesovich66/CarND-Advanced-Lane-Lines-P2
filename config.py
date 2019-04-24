
# System settings
video_mode = 0  # =0: test images, =1 video playback
store_img = 0  # =1: store intermediate images in test images mode
plot_figures = 1  # if set =1 then plots figure at different stages of the pipeline

# Video file to test
video_file_name = 'project_video'
clip_video = 1  # [default =0] Full video playback
                # [= 1] video file will be clipped with the following 2 parameters
clip_start = 20  # presenting the start of the subclip, used when clip_video = 1
clip_end = 24  # presenting the start of the subclip, used when clip_video = 1

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
algo_version = 0  #  S color channel (HLS) && Sobel x, gradient threshold
s_thresh = (255, 255)  # Threshold S color channel (HLS)

# Gradient algos
sobel_kernel_size = 23
sxy_thresh = (20, 150)  # Sobel x or y, gradient threshold
mag_thresh = (0, 255)  # gradient magnitude threshold
abs_grad_thresh = (0, np.pi/2)  # Absolute value of the gradient direction

