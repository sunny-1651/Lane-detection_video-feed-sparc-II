[//]: # (Image References)
[image1]: ./output_images/Undistorted_Image.jpg "ChessBoard Undistorted Example"
[image2]: ./output_images/Real_Undistorted_Image.jpg "Example of Distortion corrected image"
[image3]: ./output_images/Binary_Image.jpg "Thresholding Example"
[image4]: ./output_images/Transformed_Image.jpg "Perspective Transform Example"
[image5]: ./output_images/lane_lines.jpg "Histogram Plot"
[image6]: ./output_images/lane_lines.jpg "Fitted Lane Line Example"
[image7]: ./output_images/test4.jpg "Result Example"
[video1]: ./test_videos_output/project_video.mp4 "Video"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images:
![alt text][image2]
```
def cal_undistort(image,cal_file='camera_cal/dist_pickle.p'):
    # Use cv2.calibrateCamera() and cv2.undistort()
    dist_pickle = pickle.load( open( cal_file, "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(image,mtx,dist,None,mtx)  
    return undist
```

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`. The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src=np.float32([[200,715],[1150,715],[620,450],[725,450]]) 
dst=np.float32([[280,715], [950,715],[280,0], [950,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 715      | 280, 715      | 
| 1150, 715     | 950, 715      |
| 620, 450      | 280, 0        |
| 725, 450      | 950, 0        |

I verified that my perspective transform onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First I take a histogram of the bottom half of the image, find the starting point for the left and right lines, then implement Sliding Windows to find pixels for left and right lane, then fit a second order polynomial for left and right lane line.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

First converting image x and y values to real world space, then fit new polynomials to x,y in world space, calculate the new radius of curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `vis()`.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

This same video can also be found at: `/test_videos_output/project_video_out.mp4`

Here's a [link to my video result].(./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although the pipeline works with the road conditions like that in project video, but it will be better to check if curvature values are not haywire (as in harder challenge video). I guess the use of neural net might make our pipeline more roboustic.

The pipeline does not works well with continous changes in lane lines color ot its shape or its lighting.
