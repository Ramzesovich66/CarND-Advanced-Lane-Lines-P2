import numpy as np
# -------------------------------------------------------------------------------
#   _____           _                            _   _   _
#  / ____|         | |                          | | | | (_)
# | (___  _   _ ___| |_ ___ _ __ ___    ___  ___| |_| |_ _ _ __   __ _ ___
#  \___ \| | | / __| __/ _ \ '_ ` _ \  / __|/ _ \ __| __| | '_ \ / _` / __|
#  ____) | |_| \__ \ ||  __/ | | | | | \__ \  __/ |_| |_| | | | | (_| \__ \
# |_____/ \__, |___/\__\___|_| |_| |_| |___/\___|\__|\__|_|_| |_|\__, |___/
#          __/ |                                                  __/ |
#         |___/                                                  |___/
# -------------------------------------------------------------------------------
video_mode = 0  # [= 0]: test images, =1 video

# The following param only when [video_mode = 1]
store_video = 1 # [= 0]: processed video is played real time, press 'q' to quit video playback
                 # [= 1]: processed video stored in a file only

# Video file to test
video_file_name = '../project_video'  # '../challenge_video'  #'../project_video'

# Path to the subfolders
cam_cal_folder = '../camera_cal/'  # calibration images
test_img_folder = '../test_images/'  # test images
output_img_folder = '../output_images/'  # output images
output_video_folder = '../output_videos/'  # output videos

# The following 3 params are only when [store_video = 1]
clip_video = 0  # [default =0]: Full video playback
                # [= 1]: video file will be clipped with the following 2 parameters
clip_start = 3  # presenting the start of the subclip, used when clip_video = 1
clip_end = 4  # presenting the start of the subclip, used when clip_video = 1

# The following 2 params only when [video_mode = 0]
store_img = 0  # =1: store intermediate images in test images mode
plot_figures = 0  # if set =1 then plots figure at different stages of the pipeline

# Calibration related params
compute_calib_params = 0  # if set to 1 then calibration params recomputed, else loaded from a file
x_dim = 9  # chessboard size x axis
y_dim = 6  # chessboard size y axis

# -------------------------------------------------------------------------------
#     /\   | |                      | | | | (_)
#    /  \  | | __ _  ___    ___  ___| |_| |_ _ _ __   __ _ ___
#   / /\ \ | |/ _` |/ _ \  / __|/ _ \ __| __| | '_ \ / _` / __|
#  / ____ \| | (_| | (_) | \__ \  __/ |_| |_| | | | | (_| \__ \
# /_/    \_\_|\__, |\___/  |___/\___|\__|\__|_|_| |_|\__, |___/
#              __/ |                                  __/ |
#             |___/                                  |___/
# -------------------------------------------------------------------------------

#
num_of_frames = 10  # Averaging is done over this number of frames

# Binary image settings
algo_version = 1  #  [= 0] simple algo: S color channel (HLS) && Sobel x, gradient threshold
                  #  [= 1] as =0 but additionally performs HSV and gray
if algo_version == 0:
    s_thresh = (170, 255)  # Threshold S color channel (HLS) (170, 255) vs (70, 255)
else:
    s_thresh = (70, 255)

# Gradient algos
sobel_kernel_size = 17
sxy_thresh = (20, 200)  # Sobel x or y, gradient threshold

# Use cv2.morphologyEx with kernel size 3x3
# to closing small holes inside the foreground objects, or small black points on the object in binary image
morphologyex_on = 1

# Perspective transform settings
'''
perspective_transform_src = np.float32([[545, 460],
                                        [733, 460],
                                        [1280., 710.],
                                        [0., 710.]])

perspective_transform_dst = np.float32([[300, 0],
                                        [1000, 0],
                                        [1000, 720],
                                        [300, 720]])
'''
perspective_transform_src = np.float32([[545, 440],
                                        [733, 440],
                                        [1280., 710.],
                                        [0., 710.]])

perspective_transform_dst = np.float32([[5, 0],
                                        [1280, 0],
                                        [950, 720],
                                        [340, 720]])


# Hyperparameters for Sliding window
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 80
# Set minimum number of pixels found to recenter window
minpix = 100

apply_search_around_poly = 1
search_around_poly = 50

# Define conversions in x and y from pixels space to meters
ym_per_pix = 60 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 400  # meters per pixel in x dimension

# Thresholds to remove bad frames
th = 1000  # Checking that the lines are separated by approximately the same distance horizontally
th1 = 0.001  # Checks first poly coefficient
th2 = 1  # Checks second poly coefficient
th3 = 300  # Checks third poly coefficient
lane_max_width = 730  # we assume that the lane width won't change with time