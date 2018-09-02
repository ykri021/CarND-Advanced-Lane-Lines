
# coding: utf-8

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


# load pickled distortion matrix
with open('wide_dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]


# # 3 - Gradients and color transforms

# In[3]:


def abs_sobel_thresh(gray, orient='x', sobel_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    
    # R & G thresholds so that yellow lanes are detected well.
    rgb_thresh = (150, 255)
    r_channel = rgb[:,:,0]
    g_channel = rgb[:,:,1]
    r_condition = np.zeros_like(r_channel)
    r_condition = (r_channel > rgb_thresh[0]) & (r_channel <= rgb_thresh[1])
    g_condition = np.zeros_like(g_channel)
    g_condition = (g_channel > rgb_thresh[0]) & (g_channel <= rgb_thresh[1])
    
    # color channel thresholds
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    
    # Threshold color channel
    s_thresh = (100, 255)
    s_condition = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])
    
    
    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
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
    
    return color_combined


# # 4 - Perspective transform to rectify binary image ("birds-eye view").

# In[4]:


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


# ## 4-1 Parameter Setting

# In[5]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[17]:


example_img = cv2.imread('./test_images/test1.jpg')
example_dst = cv2.undistort(example_img, mtx, dist, None, mtx)
example_warped = get_warped_image(example_dst)
example_warped = cv2.cvtColor(example_warped, cv2.COLOR_BGR2RGB)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.imshow(example_warped)


# In[18]:


def color_select(r, g, b, h, l, s, y, u, v, h2, s2, v2,l2, a, b2, l3, u2, v3,                  min_rgb=150, max_rgb=225, min_hls=100, max_hls=255, min_yuv=138, max_yuv=255,                  min_hsv=130, max_hsv=255, min_lab=140, max_lab=200, min_luv=125, max_luv=255):
    # 1) Convert to color space
    rgb_copy = example_warped.copy()
    hls_copy = example_warped.copy()
    yuv_copy = example_warped.copy()
    hsv_copy = example_warped.copy()
    lab_copy = example_warped.copy()
    luv_copy = example_warped.copy()
    rgb = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(hls_copy, cv2.COLOR_BGR2HLS)
    yuv = cv2.cvtColor(yuv_copy, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(hsv_copy, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(lab_copy, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(luv_copy, cv2.COLOR_RGB2LUV)
    # 2) Apply a threshold
    r_channel = rgb[:,:,0]
    g_channel = rgb[:,:,1]
    b_channel = rgb[:,:,2]
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    y_channel = yuv[:,:,0]
    u_channel = yuv[:,:,1]
    v_channel = yuv[:,:,2]
    h2_channel = hsv[:,:,0]
    s2_channel = hsv[:,:,1]
    v2_channel = hsv[:,:,2]
    l2_channel = lab[:,:,0]
    a_channel = lab[:,:,1]
    b2_channel = lab[:,:,2]
    l3_channel = luv[:,:,0]
    u2_channel = luv[:,:,1]
    v3_channel = luv[:,:,2]
    r_binary_output = np.zeros_like(r_channel)
    g_binary_output = np.zeros_like(g_channel)
    b_binary_output = np.zeros_like(b_channel)
    h_binary_output = np.zeros_like(h_channel)
    l_binary_output = np.zeros_like(l_channel)
    s_binary_output = np.zeros_like(s_channel)
    y_binary_output = np.zeros_like(y_channel)
    u_binary_output = np.zeros_like(u_channel)
    v_binary_output = np.zeros_like(v_channel)
    h2_binary_output = np.zeros_like(h2_channel)
    s2_binary_output = np.zeros_like(s2_channel)
    v2_binary_output = np.zeros_like(v2_channel)
    l2_binary_output = np.zeros_like(l2_channel)
    a_binary_output = np.zeros_like(a_channel)
    b2_binary_output = np.zeros_like(b2_channel)
    l3_binary_output = np.zeros_like(l3_channel)
    u2_binary_output = np.zeros_like(u2_channel)
    v3_binary_output = np.zeros_like(v3_channel)
    binaries = []
    if r:
        r_binary_output[(r_channel > min_rgb) & (r_channel <= max_rgb)] = 1
        binaries.append(r_binary_output)
    if g:
        g_binary_output[(g_channel > min_rgb) & (g_channel <= max_rgb)] = 1
        binaries.append(g_binary_output)
    if b:
        b_binary_output[(b_channel > min_rgb) & (b_channel <= max_rgb)] = 1
        binaries.append(b_binary_output)
    if h:
        h_binary_output[(h_channel > min_hls) & (h_channel <= max_hls)] = 1
        binaries.append(h_binary_output)
    if l:
        l_binary_output[(l_channel > min_hls) & (l_channel <= max_hls)] = 1
        binaries.append(l_binary_output)
    if s:
        s_binary_output[(s_channel > min_hls) & (s_channel <= max_hls)] = 1
        binaries.append(s_binary_output)
    if y:
        y_binary_output[(y_channel > min_yuv) & (y_channel <= max_yuv)] = 1
        binaries.append(y_binary_output)
    if u:
        u_binary_output[(u_channel > min_yuv) & (u_channel <= max_yuv)] = 1
        binaries.append(u_binary_output)
    if v:
        v_binary_output[(v_channel > min_yuv) & (v_channel <= max_yuv)] = 1
        binaries.append(v_binary_output)
    if h2:
        h2_binary_output[(h2_channel > min_hsv) & (h2_channel <= max_hsv)] = 1
        binaries.append(h2_binary_output)
    if s2:
        s2_binary_output[(s2_channel > min_hsv) & (s2_channel <= max_hsv)] = 1
        binaries.append(s2_binary_output)
    if v2:
        v2_binary_output[(v2_channel > min_hsv) & (v2_channel <= max_hsv)] = 1
        binaries.append(v2_binary_output)
    if l2:
        l2_binary_output[(l2_channel > min_lab) & (l2_channel <= max_lab)] = 1
        binaries.append(l2_binary_output)
    if a:
        a_binary_output[(a_channel > min_lab) & (a_channel <= max_lab)] = 1
        binaries.append(a_binary_output)
    if b2:
        b2_binary_output[(b2_channel > min_lab) & (b2_channel <= max_lab)] = 1
        binaries.append(b2_binary_output)
    if l3:
        l3_binary_output[(l3_channel > min_luv) & (l3_channel <= max_luv)] = 1
        binaries.append(l3_binary_output)
    if u2:
        u2_binary_output[(u2_channel > min_luv) & (u2_channel <= max_luv)] = 1
        binaries.append(u2_binary_output)
    if v3:
        v3_binary_output[(v3_channel > min_luv) & (v3_channel <= max_luv)] = 1
        binaries.append(v3_binary_output)
    
    color_binary = np.zeros_like(r_channel)
    
    if len(binaries) == 0:
        return color_binary
    elif len(binaries) > 0:
        binary_exm_out = (binaries[0] == 1)
    if len(binaries) > 1:
        for l in range(1, len(binaries)):
            binary_exm_out = binary_exm_out & (binaries[l] == 1)

    color_binary[binary_exm_out] = 1
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(example_warped)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(color_binary, cmap="gray")
    ax2.set_title('Color Thresholded Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # 3) Return a binary image of threshold result
    return color_binary


# In[19]:


interact(color_select,r=False, g=False, b=False, h=False, l=False, s=True, y=False, u=False, v=False, h2=False, s2=False, v2=True,          l2=False, a=False, b2=True, l3=True, u2=False, v3=False,         min_rgb=(0,255), max_rgb=(0,255), min_hls=(0,255), max_hls=(0,255), min_yuv=(0,255), max_yuv=(0,255),         min_hsv=(0,255), max_hsv=(0,255), min_lab=(0,255), max_lab=(0,255), min_luv=(0,255), max_luv=(0,255));


# In[9]:


def update_abs_sobel(x=True, y=True, min_sobel_x=12, max_sobel_x=255, min_sobel_y=25, max_sobel_y=255):
    # Convert to grayscale
    gray = cv2.cvtColor(example_warped, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    gradx = np.zeros_like(gray)
    grady = np.zeros_like(gray)
    if x:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        gradx = np.zeros_like(scaled_sobelx)
        gradx[(scaled_sobelx >= min_sobel_x) & (scaled_sobelx <= max_sobel_x)] = 1
    if y:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobely = np.absolute(sobely)
        scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        grady = np.zeros_like(scaled_sobely)
        grady[(scaled_sobely >= min_sobel_y) & (scaled_sobely <= max_sobel_y)] = 1
    grad_binary = np.zeros_like(gray)
    if gradx.any() == 1 and grady.any() == 1:
        grad_binary = np.zeros_like(scaled_sobelx)
        grad_binary [(gradx == 1) & (grady == 1)] = 1
    elif gradx.any() != 1:
        grad_binary = np.zeros_like(scaled_sobely)
        grad_binary = grady
    elif grady.any() != 1:
        grad_binary = np.zeros_like(scaled_sobelx)
        grad_binary = gradx
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(example_warped)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(grad_binary, cmap="gray")
    ax2.set_title('abs sobel Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # Return this mask as your binary_output image
    return grad_binary


# In[10]:


interact(update_abs_sobel, x=True, y=True, min_sobel_x=(0,255), max_sobel_x=(0,255), min_sobel_y=(0,255), max_sobel_y=(0,255));


# In[11]:


def update_mag(sobel_kernel=3, min_mag=25, max_mag=255):
    gray = cv2.cvtColor(example_warped, cv2.COLOR_RGB2GRAY)
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
    mag_binary[(gradmag >= min_mag) & (gradmag <= max_mag)] = 1
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(example_warped)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(mag_binary, cmap="gray")
    ax2.set_title('Magnitude Thresholded Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # Return this mask as your binary_output image
    return mag_binary


# In[12]:


interact(update_mag, sobel_kernel=(1,31,2), min_mag=(0,255), max_mag=(0,255));


# In[13]:


def update_dir(sobel_kernel=3, min_dir=0.7, max_dir=1.3):
    gray = cv2.cvtColor(example_warped, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are met
    # Return this mask as your binary_output image
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= min_dir) & (absgraddir <= max_dir)] = 1
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(example_warped)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dir_binary, cmap="gray")
    ax2.set_title('Directory Thresholded Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return dir_binary


# In[14]:


interact(update_dir, sobel_kernel=(1,31,2), min_dir=(0,np.pi/2, 0.1), max_dir=(0,np.pi/2, 0.1));

