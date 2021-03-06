
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# 
# # Import Packages

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1 - Caliberation
# 

# In[2]:


nx = 9
ny = 6

camera_cal_img_dir = "./camera_cal/"
camera_cal_output_dir = "./camera_cal_output/"

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
camera_cal_images = glob.glob(camera_cal_img_dir + '*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(camera_cal_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        image_name = os.path.split(fname)[1]
        write_name = camera_cal_output_dir + 'cornersfound_' + image_name
        cv2.imwrite(write_name,img)
        #cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# ## test on images

# In[3]:


# Test undistortion on an image
img = cv2.imread(camera_cal_img_dir + 'calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# In[4]:


# Save the camera calibration result for later use (we won't worry about rvecs / tvecs) 
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )


# In[5]:


undistorted_dir = "undistorted_images/"

# load pickled distortion matrix
with open('wide_dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
# Visualize undistortion
# Step through the list and search for chessboard corners
for idx, fname in enumerate(camera_cal_images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    image_name = os.path.split(fname)[1]
    os.makedirs(camera_cal_output_dir + undistorted_dir, exist_ok=True)
    write_name = camera_cal_output_dir + undistorted_dir + 'undistorted_' + image_name
    cv2.imwrite(write_name,dst)
    print(write_name)
    #cv2.imshow('dst', dst)
    cv2.waitKey(500)
cv2.destroyAllWindows()


# ## 2 - Apply a distortion correction to raw images.

# In[6]:


test_images_dir = "./test_images/"
output_images_dir = "./output_images/"
undistorted_dir = 'undistorted_images/'

test_images = glob.glob(test_images_dir + '*.jpg')

for idx, fname in enumerate(test_images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    image_name=os.path.split(fname)[1]
    os.makedirs(output_images_dir + undistorted_dir, exist_ok=True)
    cv2.imwrite(output_images_dir + undistorted_dir + image_name ,dst)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image' + str(idx+1), fontsize=30)
    ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image' + str(idx+1), fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

cv2.destroyAllWindows()


# # 3 - Gradients and color transforms
# 
# I used 7 kinds of gradient thresholds:
# 
# 1. Along the X axis and Y axis.
# 2. Magnitude of the Gradient with thresholds of 25 and 255.
# 3. Directional gradient with thresholds of 0.7 and 1.3 degrees.
# 4. S channel threshold from HLS color space since it picks up the yellow lines well and white lines a little.
# 5. V channel threshold from HSV color space since it picks up the yellow lines and white lines well but on sunny roads, it does not detect lines.
# 6. L channel threshold from LUV color space since it picks up the yellow lines and white lines well but on sunny roads, it does not detect lines.
# 7. L channel threshold from LAB color space since it picks up the yellow lines well.

# In[7]:


def abs_sobel_thresh(gray, orient='x', sobel_thresh=(0, 255)):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    # Return this mask as your binary_output image
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return this mask as your binary_output image
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are met
    # Return this mask as your binary_output image
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    return dir_binary

def get_thresholded_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    
    gradx = abs_sobel_thresh(gray, orient='x', sobel_thresh=(12, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_thresh=(25, 255))

    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(25, 255))
    
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, dir_thresh=(0.7, 1.3))
    
    # combine the gradient , magnitude thresolds and direction thresholds.
    combined_condition = ((gradx == 1) & (grady == 1) | (mag_binary == 1) & (dir_binary == 1))
    
    # color channel thresholds
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    
    # Threshold color channel
    s_thresh = (100, 255)
    s_condition = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])    
    
    l_thresh = (125, 255)
    l_condition = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])
    
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_thresh = (130, 255)
    v_condition = (v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])
    
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    b2_channel = lab[:,:,2]
    b2_thresh = (145,200)
    b2_condition = (b2_channel > b2_thresh[0]) & (b2_channel <= b2_thresh[1])
    
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l2_channel = lab[:,:,0]
    l2_thresh = (125,255)
    l2_condition = (l2_channel > l2_thresh[0]) & (l2_channel <= l2_thresh[1])
    
    
    color_combined = np.zeros_like(s_channel)
    color_combined[(s_condition & v_condition) | (l2_condition & b2_condition) | combined_condition] = 1
    
    # apply ROI
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(color_combined, mask)
    
    return thresholded


# ## Test on Images

# In[8]:


thresholded_dir = 'thresholded_images/'
undistorted_images = glob.glob(output_images_dir + undistorted_dir + '*.jpg')

for idx, fname in enumerate(undistorted_images):
    img = cv2.imread(fname)
    thresholded = get_thresholded_image(img)
    image_name=os.path.split(fname)[1]
    os.makedirs(output_images_dir + thresholded_dir, exist_ok=True)
    cv2.imwrite(output_images_dir + thresholded_dir + image_name ,thresholded)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Undistorted Image' + str(idx+1), fontsize=30)
    ax2.imshow(thresholded, cmap='gray')
    ax2.set_title('Thresholded Image' + str(idx+1), fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

cv2.destroyAllWindows()


# # 4 - Perspective transform to rectify binary image ("birds-eye view").

# In[9]:


# Define perspective transform function
def get_warped_image(img, trans=True):
    
    # Define calibration box in source (original) and destination (desired and warped) coordinates
    img_size = (img.shape[1], img.shape[0])
    
    # Four source coordinates
    bottom_left = [220,720]
    bottom_right = [1110, 720]
    top_left = [570, 470]
    top_right = [722, 470]
    src = np.float32([top_left,bottom_left,bottom_right,top_right])
    

    pts = np.array([top_left,bottom_left,bottom_right,top_right], np.int32)
    pts = pts.reshape((-1,1,2))
    copy = img.copy()
    cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)
    
    # Four desired coordinates
    bottom_left = [320,720]
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]
    

    dst = np.float32([top_left,bottom_left,bottom_right,top_right])
    
    if trans:
        # compute the perspective transform M
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        # Could compute the inverse also by swaping the input parameters
        M = cv2.getPerspectiveTransform(dst, src)
    
    # Create warped image - uses linear interpotation
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)
    
    return warped


# ## Test on Images

# In[10]:


original_images = glob.glob(test_images_dir + '*.jpg')
warped_dir = 'warped_images/'

for idx, fname in enumerate(original_images):
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    thresholded = get_thresholded_image(dst)
    warped = get_warped_image(thresholded)
    image_name=os.path.split(fname)[1]
    os.makedirs(output_images_dir + warped_dir, exist_ok=True)
    cv2.imwrite(output_images_dir + warped_dir + image_name ,warped)
    
    # Four source coordinates
    bottom_left = [220,720]
    bottom_right = [1110, 720]
    top_left = [570, 470]
    top_right = [722, 470]

    src = np.float32([bottom_left,bottom_right,top_right,top_left])
    pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
    pts = pts.reshape((-1,1,2))
    copy = img.copy()
    cv2.polylines(copy,[pts],True,(0,0,255), thickness=3)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image' + str(idx+1), fontsize=30)
    ax2.imshow(warped, cmap="gray")
    ax2.set_title('Warped Image' + str(idx+1), fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

cv2.destroyAllWindows()


# # 5 - Detect lane pixels and fit to find the lane boundary.
# 
# ## 5-1 Finding the Lines: Histogram Peaks

# In[11]:


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    return histogram


# In[12]:


img = mpimg.imread('test_images/project2.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = get_thresholded_image(dst)
warped = get_warped_image(thresholded)
histogram = hist(warped)
plt.plot(histogram)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# ## 5-2 Finding the Lines: Sliding Window

# In[13]:


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


# In[14]:


img = mpimg.imread('test_images/project2.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = get_thresholded_image(dst)
warped = get_warped_image(thresholded)

out_img = fit_polynomial(warped)

plt.imshow(out_img)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# ## 5-3 Finding the Lines: Search from Prior

# In[15]:


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    left_fit=None
    right_fit=None

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    if (left_fit is None or right_fit is None):
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
      
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result, left_fitx, right_fitx, ploty


# In[16]:


img = mpimg.imread('test_images/project2.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = get_thresholded_image(dst)
warped = get_warped_image(thresholded)

result, left_fitx, right_fitx, ploty = search_around_poly(warped)

plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, warped.shape[1])
plt.ylim(warped.shape[0],0)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # 6 - Determine the curvature of the lane and vehicle position with respect to center.

# In[17]:


def measure_radius_of_curvature(x_values, img_length):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # If no pixels were found return None
    y_points = np.linspace(0, img_length-1, img_length)
    y_eval = np.max(y_points)
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


# In[18]:


img = mpimg.imread('test_images/test6.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = get_thresholded_image(dst)
warped = get_warped_image(thresholded)

result, left_fitx, right_fitx, ploty = search_around_poly(warped)

left_curve_rad = measure_radius_of_curvature(left_fitx, warped.shape[0])
right_curve_rad = measure_radius_of_curvature(right_fitx, warped.shape[0])
average_curve_rad = (left_curve_rad + right_curve_rad)/2
curvature_string = "Radius of curvature: %.2f m" % average_curve_rad
print(curvature_string)

# compute the offset from the center
lane_center = (right_fitx[719] + left_fitx[719])/2
xm_per_pix = 3.7/700 # meters per pixel in x dimension
center_offset_pixels = lane_center - img_size[0]/2
center_offset_mtrs = xm_per_pix*center_offset_pixels
offset_string = "Center offset: %.2f m" % center_offset_mtrs
print(offset_string)


# # 7 - Warp the detected lane boundaries back onto the original image.

# In[19]:


def draw_poly(img, warped, left_fitx, right_fitx):
    out_img = np.dstack((warped, warped, warped))*255
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    
    left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
    line_points = np.vstack([left_line_window, right_line_window])
    
    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
    unwarped= get_warped_image(out_img, trans=False)
    draw_result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
    
    return draw_result


# In[20]:


img = mpimg.imread('test_images/test6.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = get_thresholded_image(dst)
warped = get_warped_image(thresholded)
poly_img, left_fitx, right_fitx, ploty = search_around_poly(warped)

draw_result = draw_poly(img, warped, left_fitx, right_fitx)
plt.imshow(draw_result)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # 8 - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# In[21]:


def fit_polynomial_final(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, minimap = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return lefty, leftx, righty, rightx, left_fitx, right_fitx, ploty, minimap


# In[22]:


def drawMiniMap(img, warped):
    margin = 100
    lefty, leftx, righty, rightx, left_fitx, right_fitx, ploty, minimap = fit_polynomial_final(warped)

    if not(leftx.shape == 0 or lefty.shape == 0 or rightx.shape == 0 or righty.shape == 0):   
        ## Visualization ##
        # Colors in the left and right lane regions
        minimap[lefty, leftx] = [255, 0, 0]
        minimap[righty, rightx] = [0, 0, 255]
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        
        # Draw the lane onto the minimap
        window_img = np.zeros_like(minimap)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0 ,255, 0))
        minimap = cv2.addWeighted(minimap, 1, window_img, 0.3, 0)
    
    
    cv2.polylines(minimap, np.int_([pts_left]), False, (255, 0, 0), thickness=10)
    cv2.polylines(minimap, np.int_([pts_right]), False, (0, 0, 255), thickness=10)
    minimap = cv2.resize(minimap,(int(0.3*minimap.shape[1]), int(0.3*minimap.shape[0])))
    
    x_offset=890
    y_offset=10
    
    img[y_offset:y_offset+minimap.shape[0], x_offset:x_offset+minimap.shape[1]] = minimap
    return img


# In[23]:


def check_mean_distance(left_fitx, right_fitx, Isbadlanes, running_mean_distance):
    mean_distance = np.mean(right_fitx - left_fitx)  
    if running_mean_distance == 0:
        running_mean_distance = mean_distance
    if (np.abs(mean_distance) < 0.9*np.abs(running_mean_distance) or np.abs(mean_distance) > 1.1*np.abs(running_mean_distance)):
        Isbadlanes = True

    return Isbadlanes, running_mean_distance


# In[24]:


def check_curveture(left_fitx, right_fitx, Isbadlanes, img_length, past_left_curve_rad, past_right_curve_rad):
    left_curve_rad = measure_radius_of_curvature(left_fitx, img_length)
    right_curve_rad = measure_radius_of_curvature(right_fitx, img_length)
    
    if(past_left_curve_rad == 0 or past_right_curve_rad == 0):
        past_left_curve_rad = left_curve_rad
        past_right_curve_rad = right_curve_rad
    elif (left_curve_rad < 0.7*past_left_curve_rad or  left_curve_rad>1.3*past_left_curve_rad) or     (right_curve_rad < 0.7*past_right_curve_rad or  right_curve_rad>1.3*past_right_curve_rad):
        Isbadlanes = True
    
    return Isbadlanes, past_left_curve_rad, past_right_curve_rad


# In[25]:


def sanity_check(left_fitx, right_fitx, Isbadlanes, running_mean_distance, img_length, past_left_curve_rad, past_right_curve_rad):
    # lane width check
    if(left_fitx is None or right_fitx is None):
        Isbadlanes = True
    else:
        Isbadlanes, running_mean_distance  = check_mean_distance(left_fitx, right_fitx, Isbadlanes, running_mean_distance)
            
    # detact lane curveture
        Isbadlanes, past_left_curve_rad, past_right_curve_rad = check_curveture(left_fitx, right_fitx, Isbadlanes, img_length,                                                                                      past_left_curve_rad, past_right_curve_rad)
    
    return Isbadlanes, running_mean_distance, past_left_curve_rad, past_right_curve_rad


# In[26]:


def get_averaged_line(past_good_lines, new_line):
    # Number of frames to average over
    num_frames = 10
    
    if new_line is None:
        pass
        if len(past_good_lines) == 0:
            # If there are no previous lines, return None
            return past_good_lines, None
        else:
            # Else return the last line
            return past_good_lines, past_good_lines[-1]
    else:
        if len(past_good_lines) < num_frames:
            # we need at least num_frames frames to average over
            past_good_lines.append(new_line)
            return past_good_lines, new_line
        else:
            # average over the last num_frames frames
            past_good_lines[0:num_frames-1] = past_good_lines[1:]
            past_good_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += past_good_lines[i]
            new_line /= num_frames
    return past_good_lines, new_line


# In[27]:


def search_lane(warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty,_ = find_lane_pixels(warped)
    
    # If no pixels were found return None
    if(leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0):
        return None, None

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fitx, right_fitx


# In[28]:


def search_around_poly_final(warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50
    left_fit = None
    right_fit = None
    past_good_right_lines = []
    past_good_left_lines = []
    running_mean_distance = 0
    past_left_curve_rad = 0
    past_right_curve_rad = 0
    

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    if (left_fit is None or right_fit is None) :
        left_fitx, right_fitx = search_lane(warped)
        
    else:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit new polynomials
        left_fitx, right_fitx,_ = fit_poly(warped.shape, leftx, lefty, rightx, righty)
    
    ## sanity check
    Isbadlanes = False
    Isbadlanes, running_mean_distance, past_left_curve_rad, past_right_curve_rad =     sanity_check(left_fitx, right_fitx, Isbadlanes, running_mean_distance, warped.shape[0], past_left_curve_rad, past_right_curve_rad)
    if Isbadlanes:
        if (len(past_good_left_lines) == 0 or len(past_good_right_lines) == 0):
            left_fitx, right_fitx = search_lane(warped)
        else:
            print("Bad Lanes")
            left_fitx = past_good_left_lines[-1]
            right_fitx = past_good_right_lines[-1]
    else:
        past_good_left_lines, left_fitx = get_averaged_line(past_good_left_lines, left_fitx)
        past_good_right_lines, right_fitx = get_averaged_line(past_good_right_lines, right_fitx)
        mean_distance = np.mean(right_fitx - left_fitx)
        running_mean_distance = 0.9 * running_mean_distance + 0.1 * mean_distance
    
    
    return left_fitx, right_fitx


# In[29]:


def pipeline(input_img):
    input_shape_h = input_img.shape[0]
    input_shape_w = input_img.shape[1]

    undst_img = cv2.undistort(input_img, mtx, dist, None, mtx)
    # get thresholded image
    thresholded_img = get_thresholded_image(undst_img)  
    # perform a perspective transform
    warped_img = get_warped_image(thresholded_img)
    left_fitx,right_fitx = search_around_poly_final(warped_img)
    draw_img = draw_poly(undst_img, warped_img, left_fitx, right_fitx)
    output_img = drawMiniMap(draw_img, warped_img)
    
    # compute the radius of curvature
    left_curve_rad = measure_radius_of_curvature(left_fitx, input_shape_h)
    right_curve_rad = measure_radius_of_curvature(right_fitx, input_shape_h)
    # average_curve_rad = (left_curve_rad + right_curve_rad)/2
    curvature_string = "Radius of curvature:"
    curvature_string_left = "left: %.2f m," % left_curve_rad
    curvature_string_right = "right: %.2f m" % right_curve_rad
    # curvature_string = "Radius of curvature: %.2f m" % average_curve_rad
    
    # compute the offset from the center
    lane_center = (right_fitx[input_shape_h-1] + left_fitx[input_shape_h-1])/2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    center_offset_pixels = lane_center - input_shape_w/2
    center_offset_mtrs = xm_per_pix*center_offset_pixels
    offset_string = "Center offset: %.2f m" % center_offset_mtrs
    
    cv2.putText(output_img,curvature_string , (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    cv2.putText(output_img,curvature_string_left , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), thickness=2)
    cv2.putText(output_img,curvature_string_right , (490, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), thickness=2)

    cv2.putText(output_img, offset_string, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)

    return output_img


# ## Test on Images

# In[30]:


output_images_dir = "./output_images/"
processed_dir = 'processed_images/'

original_images = glob.glob(test_images_dir + '*.jpg')

for idx, fname in enumerate(original_images):
    img = cv2.imread(fname)
    # Apply pipeline
    processed = pipeline(img)
    image_name=os.path.split(fname)[1]
    os.makedirs(output_images_dir + processed_dir, exist_ok=True)
    cv2.imwrite(output_images_dir + processed_dir + image_name ,processed)
    
    # Plot the 2 images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image'+ str(idx+1), fontsize=30)
    ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    ax2.set_title('Processed Image'+ str(idx+1), fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

cv2.destroyAllWindows()


# ## Test on Videos

# In[31]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[32]:


output = 'project_video_output.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(25,30)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(output, audio=False)')


# In[33]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))


# In[34]:


challenge_output = 'challenge_video_output.mp4'

#clip2 = VideoFileClip("challenge_video.mp4").subclip(0,5)
clip2 = VideoFileClip("challenge_video.mp4")
challenge_clip = clip2.fl_image(pipeline) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[35]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[36]:


harder_challenge_output = 'harder_challenge_video_output.mp4'
clip3 = VideoFileClip("harder_challenge_video.mp4")
harder_challenge_clip = clip3.fl_image(pipeline) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'harder_challenge_clip.write_videofile(harder_challenge_output, audio=False)')


# In[37]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(harder_challenge_output))

