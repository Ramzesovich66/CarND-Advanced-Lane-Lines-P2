import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import pickle
import matplotlib.pyplot as plt
from plot_func import plot_images

# Import configuration parameters
import config as cfg


def camera_calibration():
    # Make a list of calibration images
    images = glob.glob(cfg.cam_cal_folder + 'calib*.jpg')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cfg.x_dim * cfg.y_dim, 3), np.float32)
    objp[:, :2] = np.mgrid[0: cfg.x_dim, 0: cfg.y_dim].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cfg.x_dim, cfg.y_dim), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cfg.x_dim, cfg.y_dim), corners, ret)
            if cfg.store_img:
                write_name = cfg.output_img_folder + 'chessboard_with_corners_img' + str(idx) + '.jpg'
                cv2.imwrite(write_name, img)
        else:
            print(fname + ' failed in finding corners')

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # store into  file mtx, dist ....
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    pickle.dump(dist_pickle, open("calibration_data.p", "wb"))

# Performs image distortion correction
def distortion_correction(dist_img):
    dist_pickle = pickle.load(open("calibration_data.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist_img = cv2.undistort(dist_img, mtx, dist, None, mtx)

    return undist_img


# Main function
if __name__ == '__main__':

    images = glob.glob(cfg.cam_cal_folder + '*.jpg')
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        undist = distortion_correction(img)
        write_name = cfg.output_img_folder + 'chessboard_undist_img' + str(idx) + '.jpg'
        #cv2.imwrite(write_name, undist)
        plot_images(img, undist, 0, 'Original Image', 'Undistored Image', 20, 'chessboard_undist_img'+str(idx))
