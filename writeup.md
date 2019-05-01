

**Advanced Lane Finding Project**
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/chessboard_with_corners_img11.jpg "Finding corners"
[image1]: ./output_images/chessboard_undist_img0.jpg "Undistorted"
[image2]: ./output_images/chessboard_undist_img11.jpg "Undistorted"
[image3]: ./output_images/test6_undist.jpg "Road Transformed"
[image4]: ./output_images/test2_undist.jpg "Road Transformed"
[image5]: ./output_images/gradient_img.jpeg "Sobel x"
[image6]: ./output_images/extracting_white_line.jpeg "Extracting white line" 
[image7]: ./output_images/extracting_yelow_line.jpeg "Extracting yellow line" 
[image8]: ./output_images/extracting_yelow_line_final.jpeg "Extracting yellow line" 
[image9]: ./output_images/final_binary_image.jpeg "Final binary image" 
[image10]: ./output_images/Bird-view_ex1.jpg "Bird view" 
[image11]: ./output_images/Bird-view_ex2.jpg "Bird view" 
[image12]: ./output_images/test2_binary_bird_view.jpg "Bird view" 
[image13]: ./output_images/straight_lines2_binary_bird_view.jpg "Bird view" 
[image14]: ./output_images/hist_on_binary_img.jpeg "Histogram" 
[image15]: ./output_images/curvature_formula.JPG "Curvature" 
[image16]: ./output_images/poly_curve_formula.JPG "Second order poly" 
[image17]: ./output_images/warped_image_final.jpeg "Final result" 
[video1]: ./output_videos/project_video.mp4 "Video"


---


### Camera Calibration

The code for this step is contained in the `"./source_code/camera_calibration.py"`.

As opposed to pinhole cameras, cameras with lenses introduce a lot of distortion to an image when mapping 3D object into 2D plane. 
Main distortions are radial (i.e. straight lines will appear curved) and tangential (i.e. when a camera lense is not parallel to a film or a sensor plane).

To removed these distortions I need to find all required calibration and distortion coefficients, 
for that purpose chess board images were used due to their high contrast and well defined pattern. 
First, I define 3D object points (`objpoints`), i.e. (x, y, z) coordinates of the chessboard corners. 
For simplicity, it was assumed that the chessboard is fixed on the (x, y) plane at z=0, 
such that the object points are the same for each calibration image. Then I used OpenCV function `findChessboardCorners()` 
to automatically find the chessboard corners (`imgpoints`):

![alt text][image0]

Knowing chessboard corners on an image and theirs coordinates in a 3D world I used another 
OpenCV function `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)` 
to compute calibration and distortion coefficients.

Then these coefficients were applied to remove distortions by using the `undist = cv2.undistort(dist, mtx, dist, None, mtx)` function and obtaining this results:
 
![alt text][image1]
![alt text][image2]
### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

Distortion correction that was calculated via camera calibration is applied to each image `undist = cv2.undistort(dist, mtx, dist, None, mtx)`:

![alt text][image3]
![alt text][image4]
#### 2. Color transforms and gradients used to create a thresholded binary image.
The code for this step is contained in the `"./source_code/binary_image.py"`.

The main idea at this step of processing is to correctly detect lane lines. We have yellow and white lane lines.
Compared to the first lane finding project, this one has lots of difficulties: shadows from guardrail, from trees, 
the road itself has lots of spots with very dark or very bright areas and so on. 
This all increases complexity of the image processing algorithms.
Here I tried different combinations of color spaces (HSV, HSL and RGB) as well as different variations of gradient techniques.

To detect white line I tried all gradient techniques that was discussed during the lecture. While they were quite good at 
detecting both lines, they also detected a lots of lines caused by shadows which was ok for a while but start causing a problem once 
I increased field of view, i.e. I modified perspective transform to "see" further to the horizon.

![alt text][image5]

Line 56 in the source code:
```python
# Convert an image into HLS color space
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# Separate the L, and S channels
l_channel = hls[:, :, 1]

# Calculate directional gradient in order to extract white line (step 1.1)
gradx = abs_sobel_thresh(l_channel, orient='x')
```
Converting an image into grayscale followed by thresholding at line 73 performed surprisingly good at extracting white line,
so I kept it instead:

![alt text][image6]
```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, thresh=210, maxval=255, type=cv2.THRESH_BINARY)
combined_white = np.zeros_like(gradx)
combined_white[((binary == 255))] = 1
```
As one can see, this also detects clearly a second white line to the right and potentially may cause a
 "jump" one more lane to the right. This, however can be easily 
avoided with "search from prior" concept or constantly monitoring detected lane width and discarding
the "bad" frame. 

For extracting a yellow line I convert an image into HSV color space at line 80 and did color thresholding:
```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_min_yellow = np.all(hsv > np.array([0, 100, 100]), axis=2)
hsv_max_yellow = np.all(hsv < np.array([40, 255, 255]), axis=2)
hsv_yellow_bin = hsv_min_yellow & hsv_max_yellow
combined_yellow = np.zeros_like(gradx)
combined_yellow[(hsv_yellow_bin == 1)] = 1
```
Which would produce:

![alt text][image7]

But for final implementation I kept just:

```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_min_yellow = np.all(hsv > np.array([0, 100, 100]), axis=2)
combined_yellow = np.zeros_like(gradx)
combined_yellow[(hsv_min_yellow == 1)] = 1
```
![alt text][image8]

Where the upper part will be successfully filtered by perspective transform. And finally we get:

![alt text][image9]

#### 3. Perspective transform

The code for this step is contained in the `"./source_code/binary_image.py"`.

The code for my perspective transform includes a function called `warper()`, which appears in lines 32 through 39.
It computes the perspective transform, M, given the source and destination points as cv2.getPerspectiveTransform(src,dst)
and warps the image using the perspective transform cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR).  
 
I chose to hardcode the source and destination points that resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545, 440      | 5, 0          | 
| 733, 440      | 1280, 0       |
| 1280, 710     | 950, 720      |
| 0, 710        | 340, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
 and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]
![alt text][image11]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

First I compute histogram of lower part of the binary image (lines 25-32 of `"./source_code/curve_computation.py"`), 
the two maximum peaks would be my lane lines center positions:

![alt text][image14]

Once, line centers found I use a sliding window, placed around the line centers, and follow the lines up to the top of the frame
accumulating all the pixels on the way (lines 35-94 of `"./source_code/curve_computation.py"`):

![alt text][image12]
![alt text][image13]

Where the green boxes are the sliding windows, red and blue pixels are the pixels detected. Having ``x`` and 
``y`` positions of each pixel I fit a second order polynomial to each set of pixels (red and blue) using ``np.polyfit``
(lines 105-112 of `"./source_code/curve_computation.py"`), the end result is the yellow curves on the above pictures. 
#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once I fitted a second order polynomial curve:

![alt text][image16]

The radius of curvature at any point ``x``  of the function ``x=f(y)``is given as 

![alt text][image15]

 For the position of the vehicle, I assumed the camera is mounted at the center of the car and the deviation of the
 midpoint of the lane from the center of the image is the offset I was looking for.
 
I did this in lines 162 through 184 in my code in `"./source_code/curve_computation.py"`

#### 6. An example image of my result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 12 through 55 in my code in `plot_func.py` in the function `annotate_frame()`.  Here is an example of my result on a test image:

![alt text][image17]

---

### Pipeline

#### 1. My link to the final video output. 

Here's a [link to my video result "./output_videos/project_video.mp4"](./output_videos/project_video.mp4)

---

### Discussion

I as I already wrote above, I extented my original destination and source look-up table for perspective transform as I was not
satisfied with the accuracy of curvature radius estimated.
The advantage of this look-up table is that we could "see" a lot further away (at some frames it was even the end of
visible road), thus, getting enough information to compute precisely curvature of the road. On the other side, this 
introduced some complexity to the processing of the image, for example, histogram based lane line center finding would
fail when lane lines from neighbour lanes detected, or would create at times some wobbly lines at the far field. To 
combat these side-effects I had to use additional pre-filtering of "bad" frames (monitoring width of the lane, checking 
whether the lines are roughly parallel, comparing poly coefficients from the current frame against long-term 
coefficients and so on), used "search from prior" concept to avoid some disadvantages of pure histogram approach,
additionally, I used circular buffer to perform long-term filtering of poly coefficients and poly curves (all this 
could be found at lines 117-126, 237-246 and 146-159 of `"./source_code/curve_computation.py"`).

I had no time to implement a lose track condition. 

It would be great to implement a dynamic and automated way to define the look-up table for perspective transform

I think gradient and color Thresholding still could be improved a lot.

Would refactor my code a bit better.