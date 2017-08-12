
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## 1. Use chessboard to calibrate the camera


```python
import numpy as np
import cv2
import glob
import helper
import matplotlib.pyplot as plt
%matplotlib inline

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

images = glob.glob('camera_cal/calibration*.jpg')

for index, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        fig = plt.figure(figsize=(18,8))
        plt.subplot(121),plt.imshow(gray,'gray'),plt.title('ORIGINAL')
        plt.subplot(122),plt.imshow(img,'gray'),plt.title('UNDISTORTED')
        
        result = img
        write_name = './camera_cal/corners_'+str(index)+'.jpg'
        cv2.imwrite(write_name,result)
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('saved corner files')
```

    saved corner files



![png](output_1_1.png)



![png](output_1_2.png)



![png](output_1_3.png)



![png](output_1_4.png)



![png](output_1_5.png)



![png](output_1_6.png)



![png](output_1_7.png)



![png](output_1_8.png)



![png](output_1_9.png)



![png](output_1_10.png)



![png](output_1_11.png)



![png](output_1_12.png)



![png](output_1_13.png)



![png](output_1_14.png)



![png](output_1_15.png)



![png](output_1_16.png)



![png](output_1_17.png)


## 2. Use the camera calibration to correct chessboard distortion


```python
img = cv2.imread('camera_cal/calibration1.jpg')
dst = helper.undistort(img, mtx, dist)
fig = plt.figure(figsize=(18,8))
plt.subplot(121),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('UNDISTORTED')

cv2.imwrite('./camera_cal/undistorted_chessboard.jpg', dst)
print('undistorted chessboard saved')
```

    undistorted chessboard saved



![png](output_3_1.png)


## 3. Show distortion-corrected test images


```python
# Make a list of test images
images = glob.glob('test_images/test*.jpg')

for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = helper.undistort(img, mtx, dist)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(dst),plt.title('UNDISTORTED')
    
    result = dst
    write_name = './test_images/corrected_'+str(index)+'.jpg'
    cv2.imwrite(write_name,result)
        
print('saved corrected test images')
```

    saved corrected test images



![png](output_5_1.png)



![png](output_5_2.png)



![png](output_5_3.png)



![png](output_5_4.png)



![png](output_5_5.png)



![png](output_5_6.png)


## Show distortion-corrected straight-lines images


```python
# Make a list of test images
images = glob.glob('test_images/straight_lines*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = helper.undistort(img, mtx, dist)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(dst),plt.title('UNDISTORTED')
```


![png](output_7_0.png)



![png](output_7_1.png)


## 5. Warp straight-lines images


```python
fig = plt.figure(figsize=(18,8))
plt.imshow(dst)
plt.plot(550, 460, '.')
plt.plot(730, 460, '.')
plt.plot(1280, 700, '.')
plt.plot(0, 700, '.')
```




    [<matplotlib.lines.Line2D at 0x11c94f940>]




![png](output_9_1.png)


## Show perspective-transformed straight-lines images


```python
images = glob.glob('test_images/straight_lines*.jpg')

for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = helper.undistort(img, mtx, dist)
    dst,_ = helper.warp(dst)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(dst),plt.title('UNDISTORTED & WARPED')
    
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    result = dst
    write_name = './test_images/perspective_transformed_corrected_'+str(index)+'.jpg'
    cv2.imwrite(write_name,result)
        
print('saved perspective transformed straight lines')
```

    saved perspective transformed straight lines



![png](output_11_1.png)



![png](output_11_2.png)


## Show perspective-transformed test images


```python
# Make a list of test images
images = glob.glob('test_images/test*.jpg')
for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = helper.undistort(img, mtx, dist)
    dst,_ = helper.warp(dst)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(dst),plt.title('UNDISTORTED & WARPED')
    
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    result = dst
    write_name = './test_images/perspective_transformed_warped_'+str(index)+'.jpg'
    cv2.imwrite(write_name,result)
        
print('saved perspective transformed test images')
```

    saved perspective transformed test images



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)



![png](output_13_5.png)



![png](output_13_6.png)


## 6. Create thresholded binary images: based on colorspace channels


```python
images = glob.glob('test_images/test*.jpg')

for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = helper.undistort(img, mtx, dist)
    img,_ = helper.warp(img)
    dst1,dst2,dst3 = helper.color_thred(img)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(141),plt.imshow(img),plt.title('WARPED')
    plt.subplot(142),plt.imshow(dst1,'gray'),plt.title('HLS - L')
    plt.subplot(143),plt.imshow(dst2,'gray'),plt.title('LAB - B')
    plt.subplot(144),plt.imshow(dst3,'gray'),plt.title('Combined')
    
    
print('saved all perspective transformed test images')
```

    saved all perspective transformed test images



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)



![png](output_15_6.png)


## Create thresholded binary images: based on Sobel gradients


```python
images = glob.glob('test_images/test*.jpg')
for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = helper.undistort(img, mtx, dist)
    img,_ = helper.warp(img)
    dst = helper.grad_thred(img)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('WARPED')
    plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Gradient x')
    
print('saved Sobel gradients')
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)


## Combine thresholded binary images based on colorspaces and gradients


```python
images = glob.glob('test_images/test*.jpg')
for index, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = helper.undistort(img, mtx, dist)
    img,_ = helper.warp(img)
    dst = helper.combine_threds(img)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img),plt.title('WARPED')
    plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Combined threshold')
    
print('saved all Combined Threshold test images')
```

    saved all Combined Threshold test images



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)


## 7. Find lane lines, and fit polynomials of lines


```python
images = glob.glob('test_images/test*.jpg')
for fname in images:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = helper.undistort(img, mtx, dist)
    img,_ = helper.warp(img)
    img = helper.combine_threds(img)
    leftx, lefty, rightx, righty = helper.locate_lanes(img)
    left_fit, right_fit = helper.fit_poly(leftx, lefty, rightx, righty)
    result, ploty, left_fitx, right_fitx = helper.fit_poly_plot(
        img, left_fit, right_fit, leftx, lefty, rightx, righty)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(121),plt.imshow(img,'gray'),plt.title('WARPED')
    plt.subplot(122),plt.imshow(result),plt.title('LANE LINES FOUND')
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
```


![png](output_21_0.png)



![png](output_21_1.png)



![png](output_21_2.png)



![png](output_21_3.png)



![png](output_21_4.png)



![png](output_21_5.png)


## 8. Project fitted lines to orginal images


```python
images = glob.glob('test_images/test*.jpg')
for fname in images:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img0 = helper.undistort(img, mtx, dist)
    img, Minv = helper.warp(img0)
    img = helper.combine_threds(img)
    leftx, lefty, rightx, righty = helper.locate_lanes(img)
    left_fit, right_fit = helper.fit_poly(leftx, lefty, rightx, righty)
    left_fit_m, right_fit_m = helper.fit_poly_m(leftx, lefty, rightx, righty)
    _, ploty, left_fitx, right_fitx = helper.fit_poly_plot(img, left_fit, right_fit, leftx, lefty, rightx, righty)
    result = helper.project_lines(img0, img, Minv, ploty, left_fitx, right_fitx)
    left_curverad, right_curverad = helper.get_curv_m(img, left_fit_m, right_fit_m)
    curverad = (left_curverad + right_curverad) / 2
    left2c, right2c = helper.dist2center_m(img, left_fit, right_fit)
    if left2c < right2c: str_side = 'left'
    else: str_side = 'right'
    fig = plt.figure(figsize=(18,8))
    font = {'family': 'sans-serif',
            'color':  'white',
            'weight': 'bold',
            'size': 'xx-large',
           }
    plt.subplot(121),plt.imshow(img0),plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(result),plt.title('PROJECTED LANES')
    plt.text(100, 50, 'Radius of Curvature = {:.0f}m'.format(curverad), fontdict=font)
    plt.text(100, 100, 'Vehicle is {:.2f}m {:s} of center'.
             format(np.absolute((right2c-left2c)/2.), str_side), fontdict=font)
```


![png](output_23_0.png)



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)



![png](output_23_4.png)



![png](output_23_5.png)


## 9. Process videos


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import line

def process_image(img):
    return helper.process_image(img, mtx, dist, left, right)
```


```python
left = line.Line()
right = line.Line()
video_output = 'test.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_image)
%time video_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video test.mp4
    [MoviePy] Writing video test.mp4


    100%|█████████▉| 1260/1261 [02:35<00:00,  9.39it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test.mp4 
    
    CPU times: user 3min 28s, sys: 36.3 s, total: 4min 5s
    Wall time: 2min 36s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="test.mp4">
</video>




## Challenge video


```python
left = line.Line()
right = line.Line()
video_output = 'test_challenge.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(process_image)
%time video_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video test_challenge.mp4
    [MoviePy] Writing video test_challenge.mp4


    100%|██████████| 485/485 [00:50<00:00,  9.60it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_challenge.mp4 
    
    CPU times: user 1min 12s, sys: 11.3 s, total: 1min 23s
    Wall time: 50.8 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="test_challenge.mp4">
</video>




## Harder challenge video


```python
left = line.Line()
right = line.Line()
video_output = 'test_harder.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
video_clip = clip1.fl_image(process_image)
%time video_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video test_harder.mp4
    [MoviePy] Writing video test_harder.mp4


    100%|█████████▉| 1199/1200 [02:45<00:00,  8.40it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_harder.mp4 
    
    CPU times: user 3min 34s, sys: 35.7 s, total: 4min 10s
    Wall time: 2min 46s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="test_harder.mp4">
</video>



