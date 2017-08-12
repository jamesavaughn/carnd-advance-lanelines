import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
import cv2
import pickle
import glob
from tracker import tracker
from moviepy.editor import VideoFileClip
from IPython.display import HTML



print ('-----Loading Calibration Data-----')
dist_pickle = pickle.load(open('camera_cal/calibration_data.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def abs_sobel_thresh(img, orient ='x', sobel_kernel=5, abs_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag_sobel = np.uint8(255 * mag_sobel/np.max(mag_sobel))
    binary_output = np.zeros_like(mag_sobel)
    binary_output[(mag_sobel >= mag_thresh[0]) & (mag_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        absgraddir = np.absolute(sobely/solbyx)
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] =1
        return binary_output

def color_threshold(img, s_thresh=(0,255),v_thresh=(0,255), h_thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height): int(img_ref.shape[0] - level * height), max(0,int(center - width)):min(int(center + width), img_ref.shape[1])] = 1
    return output

def process_image(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixel of interests
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', abs_thresh=(12,220))
    grady = abs_sobel_thresh(img, orient='y', abs_thresh=(25,220))
    c_binary = color_threshold(img, s_thresh=(100,225), v_thresh=(50,225))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] - 255

    # define region of interest, origin (x=0,y=0) is in upper lefft
    img_size = (img.shape[1], img.shape[0])
    bot_width = .75 # percent to bottom trapizoid height
    mid_width = .08 # percet of middle trapizoid height
    height_pct = .63 # percent for trapizoid height, how far down the road
    bottom_trim = .935 # percent from top to bottom to avoid car hood

#40:00
    p0 = img.shape[1] * (.5 - mid_width/2), img.shape[0] * height_pct #top left corner #.5
    p1 = img.shape[1] * (.5 + mid_width/2), img.shape[0] * height_pct #top right corner
    p2 = img.shape[1] * (.5 + bot_width/2), img.shape[0] * bottom_trim #bottom right corner
    p3 = img.shape[1] * (.5 - bot_width/2), img.shape[0] * bottom_trim #bottom left corner

    src = np.float32([[p0],[p1],[p2],[p3]])

    offset = img_size[0] * .25 # controls how small the lane is .25

    d0 = offset, 0 #top left corner
    d1 = img_size[0] - offset, 0 #top right corner
    d2 = img_size[0] - offset, img_size[1] #bottom right corner
    d3 = offset, img_size[1] #bottom left corner

    dst = np.float32([[d0],[d1],[d2],[d3]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src) #does the reverse transform on images
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags = cv2.INTER_LINEAR)

    window_width = 25
    window_height = 80

    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_xm = 10/720, My_ym = 4/384, Mysmooth_factor = 15)

    window_centroids = curve_centers.find_window_centroids(warped) #creates center points to draw lane lines

    l_points = np.zeros_like(warped) #create mask for left points
    r_points = np.zeros_like(warped) #create mask for right points

    rightx =[] #lists
    leftx = []

    #go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0],level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1],level)
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_points[(l_points == 255) | ((l_mask == 1))] = 255 #add graphic points from the window mask here to total pixels found
        r_points[(r_points == 255) | ((r_mask == 1))] = 255 #add graphic points from the window mask here to total pixels found

    # fit the lane boundaries to the left and right center positions found
    yvals = range(0,warped.shape[0])

    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals + yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals + yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis=0))),np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] + window_width / 2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis=0))),np.int32)

    road = np.zeros_like(img) #mask to draw the road
    road_bkg = np.zeros_like(img) #mask for road background
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags = cv2.INTER_LINEAR) #transforms drawn lines back to original perspective
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags = cv2.INTER_LINEAR)

    #result = road_warped
    #write_name = './test_images/Road_Warped_'+str(index)+'.jpg'
    #cv2.imwrite(write_name,result)
    #print('saved road warped files')

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, 1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dimension
    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2) #curavture of left line
    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])

    #54:44
    #result = road_warped
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50,50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + '(m)' + side_pos + 'of center', (50,100) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #write_name = './test_images/tracked' + str(index) + '.jpg'
    #cv2.imwrite(write_name, result)

    return result

Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_video, audio = False)

#print('images processed and saved')
